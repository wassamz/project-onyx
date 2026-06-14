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

// MARK: - ChatModelCatalog
//
// PURPOSE: Single Llama 3.2 1B model shown in the Models tab.

// MARK: - ChatModelFamily

/// Which model family this descriptor belongs to.
///
/// Used to pick SF Symbol icons and apply family-specific colour coding in
/// the UI. Has no effect on inference — the model's chat template handles
/// any family-specific formatting automatically.
public enum ChatModelFamily: String, Sendable, Codable, CaseIterable {
    case llama
    case qwen
    case gemma
    case other
    
    public var symbolName: String {
        switch self {
        case .llama: return "l.circle.fill"
        case .qwen: return "q.circle.fill"
        case .gemma: return "g.circle.fill"
        case .other: return "cpu"
        }
    }
}

// MARK: - ChatModelDescriptor

/// Metadata for a downloadable MLX chat model.
///
/// Every field is used by the Models tab UI. Add a new model by creating a
/// `ChatModelDescriptor` and appending it to `ChatModelCatalog.all`.
public struct ChatModelDescriptor: Sendable, Hashable, Identifiable {
    
    /// HuggingFace repo id, e.g. `"mlx-community/Qwen2.5-3B-Instruct-4bit"`.
    ///
    /// This id is also used as the directory name under the model store:
    /// `OnyxPaths.modelDirectory(for: id)`.
    public let id: String
    
    /// Short human-readable name displayed in the UI, e.g. `"Qwen 2.5 3B (4-bit)"`.
    public let displayName: String
    
    /// Model family, used for icons and colour coding.
    public let family: ChatModelFamily
    
    /// Approximate download size in bytes. Shown as "≈ 2 GB" before download
    /// starts. The actual download may differ slightly.
    public let approxSizeBytes: Int64
    
    /// Glob patterns passed to `HubApi.snapshot(matching:)`. The defaults
    /// cover all files needed by MLX models from `mlx-community`.
    public let filePatterns: [String]
    
    /// One-line description shown beneath the model name in the list.
    public let summary: String
    
    public init(id: String, displayName: String, family: ChatModelFamily,
                approxSizeBytes: Int64, filePatterns: [String], summary: String) {
        self.id = id; self.displayName = displayName; self.family = family
        self.approxSizeBytes = approxSizeBytes
        self.filePatterns = filePatterns; self.summary = summary
    }
}

// MARK: - ChatModelCatalog

/// The curated catalog of downloadable chat models.
public enum ChatModelCatalog {
    
    /// Glob patterns that cover every file type needed by 4-bit MLX models
    /// from `mlx-community`. Reused by all catalog entries.
    public nonisolated static let defaultFilePatterns: [String] = [
        "*.json",
        "*.safetensors",
        "*.txt",
        "tokenizer.model"
    ]
    
    // swiftlint:disable line_length
    /// All available models, in display order within each device class.
    ///
    /// The first entry is treated as the recommended default by the
    /// Models tab (shown with a "Recommended" badge).
    ///
    /// - Note: To add a model, append a `ChatModelDescriptor` here. All
    ///   other subsystems (downloader, registry, UI) pick it up automatically.
    public nonisolated static let all: [ChatModelDescriptor] = [
        ChatModelDescriptor(
            id: "mlx-community/Llama-3.2-1B-Instruct-4bit",
            displayName: "Llama 3.2 1B Instruct (4-bit)",
            family: .llama,
            approxSizeBytes: 858_993_459,   // ≈ 0.8 GB
            filePatterns: defaultFilePatterns,
            summary: "Meta's compact on-device model. Under 1 GB — fast load times."
        ),
        ChatModelDescriptor(
            id: "mlx-community/Llama-3.2-3B-Instruct-4bit",
            displayName: "Llama 3.2 3B Instruct (4-bit)",
            family: .llama,
            approxSizeBytes: 1_810_000_000, // ≈ 1.81 GB
            filePatterns: defaultFilePatterns,
            summary: "Meta's heavily optimized 3B mobile chat baseline. Outstanding conversational flow, structured output handling, and reasoning layout built to safely run within 8GB iOS memory limits."
        ),
        ChatModelDescriptor(
            id: "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            displayName: "Qwen 2.5 1.5B Instruct (4-bit)",
            family: .qwen,
            approxSizeBytes: 869_000_000, // ≈ 869 MB
            filePatterns: defaultFilePatterns,
            summary: "Ultra-lean general text variant from Alibaba. Sits comfortably under 1GB of memory, enabling blisteringly fast processing speeds and massive headroom for long conversation cache pools."
        ),
        ChatModelDescriptor(
            id: "mlx-community/gemma-4-e4b-it-4bit",
            displayName: "Gemma 4 E4B Instruct (4-bit)",
            family: .gemma,
            approxSizeBytes: 5_368_709_120, // ≈ 5.00 GB
            filePatterns: defaultFilePatterns,
            summary: "Google DeepMind 2026 edge model. Features 4.5B active parameters, Per-Layer Embeddings (PLE), and a native 128K context window. Still under testing as it pushes the limits of iPhone 15 memory."
        )
    ]
    // swiftlint:enable line_length
    
    /// Look up a descriptor by its HuggingFace id.
    ///
    /// Returns `nil` for unknown ids — avoids crashing when the user
    /// manually drops extra directories into the model store.
    public nonisolated static func descriptor(forId id: String) -> ChatModelDescriptor? {
        all.first { $0.id == id }
    }
}
