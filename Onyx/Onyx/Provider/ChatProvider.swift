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
import MLXLMCommon
import Metal

// MARK: - ChatProvider
//
// PURPOSE: The main interface between the SwiftUI view layer and the MLX
//          inference engine. Manages one conversation at a time.
//
// DESIGN:
//   • @MainActor — all property changes and UI-driven method calls happen on
//     the main thread. No manual DispatchQueue.main.async required.
//   • @Observable — SwiftUI views automatically re-render when `isGenerating`
//     changes. No @StateObject or .onReceive boilerplate needed.
//   • Uses actors underneath: MLXModelManager, MLXConversationHistory, and
//     ChatModelRegistry handle their own thread safety.
//
// SYSTEM PROMPT:
//   Read from UserDefaults["onyx.systemPrompt"]. Change it at runtime;
//   the updated value is used on the next `respond()` call. Clear history
//   afterwards to apply it from the first turn.
//
// LOGGING:
//   By default every outgoing prompt payload is printed to the Xcode console.
//   This makes it easy to inspect what the model actually receives (system
//   prompt, full conversation history, etc.) without a debugger.
//   Silence it: UserDefaults.standard.set(false, forKey: "onyx.logPrompts")
//
// EXTENDING THIS:
//   To add streaming progress (bytes, ETA) during model loading, subscribe
//   to ChatModelDownloader.shared.subscribe(id:) from a separate view.
//   To persist conversations across launches, encode MLXConversationHistory's
//   turns to JSON and write to OnyxPaths.baseDirectory().

// MARK: - ChatProvider

/// @MainActor view-model that bridges the SwiftUI chat UI to the MLX runtime.
///
/// ## Quick-start
/// ```swift
/// @State private var provider = ChatProvider.shared
///
/// // Send a message
/// let stream = try await provider.respond(to: userText)
/// var reply = ""
/// for await chunk in stream {
///     reply += chunk
/// }
///
/// // Cancel mid-stream
/// provider.cancel()
///
/// // Clear conversation
/// await provider.clearHistory()
/// ```
///
/// ## Prerequisites
/// 1. Download a model in the Models tab (uses `ChatModelDownloader`).
/// 2. Activate it (uses `ChatModelRegistry`).
/// 3. Call `respond(to:)` — the model loads automatically if not already resident.
@MainActor
@Observable
public final class ChatProvider {

    // MARK: - UserDefaults keys

    /// UserDefaults key for the system prompt injected before every conversation.
    ///
    /// Write to this key (or use the `systemPrompt` property) to customise the
    /// assistant's personality, language, or focus area.
    public static let systemPromptKey = "onyx.systemPrompt"

    /// UserDefaults key to enable/disable console prompt logging.
    ///
    /// Default: `true` (logging on). Set to `false` to silence debug output.
    public static let logPromptsKey = "onyx.logPrompts"

    /// The default system prompt used when no custom prompt has been set.
    ///
    /// Replace this with any role description or set of instructions that
    /// makes sense for your use case.
    public static let defaultSystemPrompt =
        "You are a helpful AI assistant. Be clear and concise."

    // MARK: - Shared instance

    /// Process-wide singleton. Use this from all views.
    public static let shared = ChatProvider()

    // MARK: - Observable state

    /// `true` while a generation stream is actively producing tokens.
    ///
    /// Bind this in the UI to:
    ///   • Show `ThinkingDotsView` while waiting for the first token.
    ///   • Swap the Send button for a Stop button during generation.
    ///   • Disable the text input field.
    public private(set) var isGenerating: Bool = false

    // MARK: - Availability

    /// `true` when a Metal GPU is available on this device.
    ///
    /// Always `true` on a real iPhone or iPad. Returns `false` in the iOS
    /// Simulator (which has no GPU). When `false`, `respond()` throws
    /// `MLXError.metalUnavailable` and the chat UI should show an
    /// "Unavailable in Simulator" banner.
    public var isAvailable: Bool {
        #if targetEnvironment(simulator)
        return false
        #else
        return MTLCreateSystemDefaultDevice() != nil
        #endif
    }

    // MARK: - System prompt

    /// The system prompt prepended to every conversation.
    ///
    /// Backed by `UserDefaults["onyx.systemPrompt"]`. Changes here take
    /// effect on the *next* `respond()` call. Call `clearHistory()` to
    /// apply a new prompt from the first turn of a fresh conversation.
    public var systemPrompt: String {
        get {
            UserDefaults.standard.string(forKey: Self.systemPromptKey)
                ?? Self.defaultSystemPrompt
        }
        set {
            UserDefaults.standard.set(newValue, forKey: Self.systemPromptKey)
        }
    }

    // MARK: - Private state

    private let history = MLXConversationHistory()
    private let manager = MLXModelManager.shared

    /// Protects `activeTask` from cross-thread mutation.
    ///
    /// `continuation.onTermination` can fire from a background thread
    /// (when the consumer deallocates the stream), so the task pointer
    /// needs explicit locking even though ChatProvider is @MainActor.
    private let activeTaskLock = NSLock()
    private var activeTask: Task<Void, Never>?

    private init() {}

    // MARK: - Public API

    /// Stream a response to `message`.
    ///
    /// Records the message in history, ensures the active model is loaded,
    /// then begins inference. Token chunks arrive via the returned stream.
    ///
    /// - Parameter message: The user's plaintext input.
    /// - Returns: `AsyncStream<String>` that yields token chunks as they
    ///   are generated. Accumulate these to build the full reply.
    /// - Throws:
    ///   - `MLXError.metalUnavailable` — running in the iOS Simulator.
    ///   - `MLXError.modelNotInstalled(id:)` — no model has been activated.
    ///   - `MLXError.modelLoadFailed(_)` — device RAM too small for model,
    ///     or the model directory is corrupted.
    ///
    /// ## Usage
    /// ```swift
    /// do {
    ///     let stream = try await ChatProvider.shared.respond(to: text)
    ///     var reply = ""
    ///     for await chunk in stream {
    ///         reply += chunk
    ///         // update UI incrementally
    ///     }
    /// } catch MLXError.modelNotInstalled {
    ///     // Show "Go to Models tab" CTA
    /// }
    /// ```
    public func respond(to message: String) async throws -> AsyncStream<String> {
        let container = try await ensureReady()
        await history.addUserMessage(message)
        return await buildGenerationStream(container: container)
    }

    /// Re-roll the last assistant response.
    ///
    /// Drops the previous assistant turn from history and runs a fresh
    /// inference pass against the same preceding user message. Use this for
    /// a "Regenerate" / "Retry" button.
    ///
    /// - Returns: A fresh token stream for the re-generated reply.
    /// - Throws: Same as `respond(to:)`.
    public func regenerateLast() async throws -> AsyncStream<String> {
        let container = try await ensureReady()
        await history.popLastAssistant()
        return await buildGenerationStream(container: container)
    }

    /// Cancel the in-flight generation.
    ///
    /// The partial response already streamed to the UI is preserved in history
    /// so the conversation remains well-formed. The `isGenerating` flag is
    /// cleared automatically when the cancelled task drains.
    ///
    /// Call this from the Stop button that replaces Send while `isGenerating`
    /// is `true`.
    public func cancel() {
        activeTaskLock.lock()
        let task = activeTask
        activeTask = nil
        activeTaskLock.unlock()
        task?.cancel()
        // isGenerating is cleared by the task body once it detects cancellation.
        // We do NOT set it to false here to avoid a race where a new generation
        // starts between cancel() and the old task's cleanup, causing the new
        // generation's isGenerating=true to be clobbered.
    }

    /// Clear the entire conversation and cancel any in-flight generation.
    ///
    /// Call this from a "New Conversation" or "Clear" button.
    public func clearHistory() async {
        cancel()
        await history.reset()
    }

    // MARK: - Status

    /// Human-readable summary of the model loader's current state.
    ///
    /// Returns one of: `"Idle"`, `"Loading…"`, `"Ready"`, `"Error: <msg>"`.
    /// Show this in the NavigationBar status dot tooltip or a debug panel.
    public func modelStatusLabel() async -> String {
        let state = await manager.state
        switch state {
        case .idle:            return "Idle"
        case .loading:         return "Loading…"
        case .ready:           return "Ready"
        case .failed(let msg): return "Error: \(String(msg.prefix(40)))"
        }
    }

    /// Number of individual turns (user + assistant) in the current history.
    ///
    /// Display as "X turns" in a context-usage indicator.
    public var turnCount: Int {
        get async { await history.turnCount }
    }

    /// Total characters currently held in history.
    ///
    /// The history drops oldest turn-pairs once this exceeds
    /// `MLXConversationHistory.defaultMaxCharacters` (16 000).
    /// Show this as a progress bar filling toward 16 000.
    public var historyCharacterCount: Int {
        get async { await history.totalCharacterCount }
    }

    // MARK: - Private: inference pipeline

    private func ensureReady() async throws -> ModelContainer {
        guard isAvailable else {
            throw MLXError.metalUnavailable
        }
        guard let activeId = await ChatModelRegistry.shared.activeId() else {
            // No model activated — show the Models tab CTA in the view.
            throw MLXError.modelNotInstalled(id: "<none selected>")
        }
        // Refuse to load if the device doesn't have enough RAM. This prevents
        // the app from being jetsam-killed on 6 GB devices by a model that
        // needs more headroom than available.
        try ChatMemoryGate.assertCanLoad(modelId: activeId)
        try await manager.ensureLoaded(modelId: activeId)
        guard let container = await manager.getContainer() else {
            throw MLXError.modelLoadFailed("Model container unavailable after loading.")
        }
        return container
    }

    private func buildGenerationStream(container: ModelContainer) async -> AsyncStream<String> {
        let prompt = systemPrompt
        let messages = await history.buildMessages(systemPrompt: prompt)

        Self.logOutgoingMessages(systemPrompt: prompt, messages: messages)

        // Attempt to start inference. If the model refuses to generate
        // (bad config, out of memory at the framework level, etc.), we still
        // need to record an assistant turn so history stays in the strict
        // [user, assistant, user, …] alternation that chat templates require.
        // Without this invariant, some models (Llama, Gemma) produce empty
        // replies on all subsequent turns.
        let tokenStream: AsyncStream<String>
        do {
            tokenStream = try await generateFromModel(
                container: container,
                messages: messages,
                modelId: await manager.currentModelId,
                maxTokens: 2048
            )
        } catch {
            let errorText = "_(Could not start generation: \(error.localizedDescription))_"
            await history.addAssistantMessage(errorText)
            clearActiveTask()
            return AsyncStream { continuation in
                continuation.yield(errorText)
                continuation.finish()
            }
        }

        isGenerating = true

        return AsyncStream<String> { [weak self] continuation in
            guard let self else { continuation.finish(); return }

            let task = Task { @MainActor [weak self] in
                guard let self else { return }

                var fullResponse = ""
                for await chunk in tokenStream {
                    if Task.isCancelled { break }
                    fullResponse += chunk
                    continuation.yield(chunk)
                }

                // Persist the response (full or partial) as the assistant turn.
                // An empty string would break chat-template parsing on the next
                // turn, so we substitute a placeholder when nothing was produced.
                let persisted = fullResponse.isEmpty
                    ? "_(no response produced)_"
                    : fullResponse
                await self.history.addAssistantMessage(persisted)

                self.isGenerating = false
                self.clearActiveTask()
                continuation.finish()
            }

            // When the consumer stops iterating (e.g. view disappears), cancel
            // the generation task so Metal threads are freed promptly.
            continuation.onTermination = { _ in task.cancel() }

            setActiveTask(task)
        }
    }

    // MARK: - Task tracking

    private func setActiveTask(_ task: Task<Void, Never>) {
        activeTaskLock.lock()
        activeTask = task
        activeTaskLock.unlock()
    }

    private func clearActiveTask() {
        activeTaskLock.lock()
        activeTask = nil
        activeTaskLock.unlock()
    }

    // MARK: - Developer logging

    /// Print the full prompt payload to the Xcode console before sending.
    ///
    /// This makes it easy to verify:
    ///   • The system prompt is exactly what you expect.
    ///   • The conversation history has been trimmed correctly.
    ///   • The user message text is clean (no accidental whitespace, etc.).
    ///
    /// Enabled by default. To silence: set `UserDefaults["onyx.logPrompts"]`
    /// to `false` in Settings or in code before calling `respond(to:)`.
    private static func logOutgoingMessages(systemPrompt: String,
                                             messages: [[String: String]]) {
        let shouldLog: Bool
        if UserDefaults.standard.object(forKey: logPromptsKey) == nil {
            shouldLog = true  // default: on
        } else {
            shouldLog = UserDefaults.standard.bool(forKey: logPromptsKey)
        }
        guard shouldLog else { return }

        let stamp = ISO8601DateFormatter().string(from: Date())
        let bar = String(repeating: "━", count: 72)

        print("\n\(bar)")
        print("📨 [Onyx] outgoing prompt — \(stamp)")
        print("\(bar)")
        print("── system prompt (\(systemPrompt.count) chars) ──")
        print(systemPrompt)
        print("")
        print("── messages (\(messages.count) turns) ──")
        for (i, m) in messages.enumerated() {
            let role    = m["role"]    ?? "?"
            let content = m["content"] ?? ""
            print("[\(i)] role=\(role)  chars=\(content.count)")
            print(String(repeating: "·", count: min(72, max(8, content.count / 4))))
            print(content.prefix(400))
            print("")
        }
        print("\(bar)\n")
    }
}
