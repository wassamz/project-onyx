// Copyright 2026 Onyx Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation
#if canImport(Metal)
import Metal
#endif

// MARK: - HardwareProfile
//
// PURPOSE: Detect the device's RAM and GPU at startup, then compute whether
//          a given model can be loaded without exhausting memory.
//
// iPhone RAM reference (as of iPhone 15/16 generation):
//   iPhone 15 base:      6 GB   → can load 3B 4-bit models (~2 GB + 2 GB headroom = 4 GB needed ✓)
//   iPhone 15 Pro/Max:   8 GB   → ✓
//   iPhone 16 base:      8 GB   → ✓
//   iPhone 16 Pro/Max:   8 GB   → ✓
//
// DETECTION METHOD: sysctl, NOT chip name. This means future iPhone models
// slot into the correct tier automatically without code changes.
//
// OVERRIDE: Set CHATM_HARDWARE_TIER=pro (or base/max/ultra) in the Xcode
// scheme's environment variables to force a tier during development.

/// Discrete hardware tier used for capability gating.
///
/// On iPhone all current devices (8 GB) resolve to `.base`. The tier
/// primarily gates model size: `canLoadModel(approxSizeBytes:)` is the
/// definitive check. Tier alone is only used for display purposes.
public enum HardwareTier: String, Sendable, CaseIterable {
    case base   // Typical iPhone: 6–8 GB RAM
    case pro    // Mac Pro / iPad Pro with 16–36 GB
    case max    // Mac Max with 32–128 GB
    case ultra  // Mac Ultra with 64–192 GB
}

/// One-shot hardware profile. Read `HardwareProfile.default` everywhere
/// rather than calling `detect()` repeatedly — it is computed once at
/// module load time (thread-safe via `static let`).
public struct HardwareProfile: Sendable {

    public let tier: HardwareTier
    public let detectedPCores: Int
    /// Physical RAM in MiB, used for memory gating without losing fractional GiB.
    public let detectedRAMMegabytes: Int
    public let detectedRAMGigabytes: Int
    public let detectedGPUMemoryMB: Int
    /// `true` when the tier was forced via the `CHATM_HARDWARE_TIER`
    /// environment variable rather than auto-detected.
    public let isEnvOverridden: Bool
    /// Human-readable chip description from `machdep.cpu.brand_string`.
    public let chipBrand: String

    // MARK: - Shared instance

    /// Process-wide hardware profile. Computed once at first access.
    public static let `default`: HardwareProfile = detect()

    // MARK: - Detection

    public static func detect() -> HardwareProfile {
        let ramBytes = readSysctlUInt64("hw.memsize") ?? 0
        let ramMB = Int(ramBytes / 1_048_576)
        let ramGB = Int(ramBytes / 1_073_741_824)
        let pCores = Int(readSysctlInt32("hw.perflevel0.physicalcpu") ?? 0)
            .nonZeroOr(ProcessInfo.processInfo.processorCount)
        let gpuMB = readGPUWorkingSetMB()
        let chipBrand = readChipBrand()

        let envValue = ProcessInfo.processInfo.environment["CHATM_HARDWARE_TIER"]?
            .trimmingCharacters(in: .whitespaces).lowercased()
        let isEnvOverridden = envValue != nil && HardwareTier(rawValue: envValue ?? "") != nil

        let tier: HardwareTier
        if let env = envValue, let forced = HardwareTier(rawValue: env) {
            tier = forced
        } else {
            tier = pickTier(pCores: pCores, ramGB: ramGB)
        }

        return HardwareProfile(tier: tier, detectedPCores: pCores,
                                detectedRAMMegabytes: ramMB, detectedRAMGigabytes: ramGB,
                                detectedGPUMemoryMB: gpuMB,
                                isEnvOverridden: isEnvOverridden, chipBrand: chipBrand)
    }

    /// Tier from (P-cores, RAM). Both axes must qualify.
    internal static func pickTier(pCores: Int, ramGB: Int) -> HardwareTier {
        if ramGB >= 64 && pCores >= 16 { return .ultra }
        if ramGB >= 32 && pCores >= 8  { return .max }
        if ramGB >= 16 && pCores >= 6  { return .pro }
        return .base
    }

    // MARK: - Per-model memory gate

    /// Default activation headroom (MB) reserved above the model's weight size.
    ///
    /// A 4-bit 3B model's runtime working set is approximately:
    ///   weights (~2 GB) + KV-cache + framework overhead ≈ 3–4 GB.
    /// The 2 GB headroom ensures the OS doesn't jetsam the app on 6 GB devices.
    public static let defaultModelHeadroomMB = 2_048

    /// Returns `true` if this device has enough physical RAM to load a model
    /// of `approxSizeBytes` without being jetsam-killed.
    ///
    /// Formula: `physicalRAM (MB) >= modelWeights (MB) + headroom (MB)`
    ///
    /// - Parameter approxSizeBytes: Model weight size in bytes (from
    ///   `ChatModelDescriptor.approxSizeBytes`).
    /// - Parameter headroomMB: Extra RAM reserved for KV-cache and overhead.
    ///   Default: 2 048 MB.
    ///
    /// ## Example
    /// ```swift
    /// let profile = HardwareProfile.default
    /// let ok = profile.canLoadModel(approxSizeBytes: 2 * 1_073_741_824)
    /// // On a 8 GB iPhone: 8192 MB >= 2048 + 2048 = 4096 → true
    /// ```
    public func canLoadModel(
        approxSizeBytes: Int64,
        headroomMB: Int = HardwareProfile.defaultModelHeadroomMB
    ) -> Bool {
        // A forced tier above `.base` is the developer escape hatch — always allow.
        if isEnvOverridden && tier != .base { return true }
        let weightsMB = Int(approxSizeBytes / 1_048_576)
        let requiredMB = weightsMB + headroomMB
        return detectedRAMMegabytes >= requiredMB
    }

    // MARK: - Diagnostics

    /// Compact summary logged at startup. Useful for debugging reports.
    public var oneLineSummary: String {
        let override = isEnvOverridden ? " [env override]" : ""
        let maxModelGB = detectedRAMGigabytes - HardwareProfile.defaultModelHeadroomMB / 1_024
        return "tier=\(tier.rawValue)\(override) chip=\(chipBrand) cores=\(detectedPCores)P RAM=\(detectedRAMGigabytes)GB GPU=\(detectedGPUMemoryMB)MB maxModel≈\(max(0,maxModelGB))GB"
    }
}

// MARK: - sysctl helpers
//
// These are marked `nonisolated` because the project-wide
// SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor would otherwise make them
// @MainActor — they must be callable from any context at startup.

/// Read a `uint64_t` sysctl by name. Returns nil on missing/zero.
nonisolated private func readSysctlUInt64(_ name: String) -> UInt64? {
    var value: UInt64 = 0
    var size = MemoryLayout<UInt64>.size
    guard sysctlbyname(name, &value, &size, nil, 0) == 0, value > 0 else { return nil }
    return value
}

/// Read an `int32_t` sysctl by name. Returns nil on failure.
nonisolated private func readSysctlInt32(_ name: String) -> Int32? {
    var value: Int32 = 0
    var size = MemoryLayout<Int32>.size
    guard sysctlbyname(name, &value, &size, nil, 0) == 0 else { return nil }
    return value
}

/// GPU `recommendedMaxWorkingSetSize` in MB. Returns 0 if Metal is unavailable.
nonisolated private func readGPUWorkingSetMB() -> Int {
#if canImport(Metal)
    guard let device = MTLCreateSystemDefaultDevice() else { return 0 }
    return Int(device.recommendedMaxWorkingSetSize / 1_048_576)
#else
    return 0
#endif
}

/// Chip brand string from `machdep.cpu.brand_string`, e.g. `"Apple A18 Pro"`.
nonisolated private func readChipBrand() -> String {
    var size = 0
    guard sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0, size > 0 else {
        return "Unknown"
    }
    var buffer = [CChar](repeating: 0, count: size)
    guard sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0) == 0 else {
        return "Unknown"
    }
    return String(cString: buffer)
}

private extension Int {
    func nonZeroOr(_ fallback: Int) -> Int { self > 0 ? self : fallback }
}
