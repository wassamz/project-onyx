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
import MLX
import MLXLLM
import MLXLMCommon
@preconcurrency import Tokenizers

// MARK: - HuggingFace Bridge
//
// MLXLMCommon's LLMModelFactory.loadContainer(from:using:) loads directly from
// a local directory URL. We only need a TokenizerLoader bridge — the Hub
// download step is handled separately by ChatModelDownloader before we get here.
//
// DEVELOPER NOTE: These are private implementation details — you should only
// ever call MLXModelManager.shared.loadModel(modelId:) from application code.

/// Bridges swift-transformers' `AutoTokenizer` to `MLXLMCommon.Tokenizer`.
private struct HubTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let upstream = try await AutoTokenizer.from(modelFolder: directory)
        return TokenizerBridge(upstream)
    }
}

/// Thread-safe shim so the swift-transformers tokenizer satisfies
/// MLXLMCommon's `Tokenizer` protocol without modifying either package.
private struct TokenizerBridge: @unchecked Sendable, MLXLMCommon.Tokenizer {
    private let upstream: any Tokenizers.Tokenizer

    init(_ upstream: any Tokenizers.Tokenizer) { self.upstream = upstream }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }
    func convertTokenToId(_ token: String) -> Int? { upstream.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { upstream.convertIdToToken(id) }
    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(messages: messages, tools: tools, additionalContext: additionalContext)
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}

// MARK: - MLXModelManager

/// Actor that owns the lifecycle of the loaded MLX language model.
///
/// ## Overview
/// Exactly one model is resident in memory at a time. The manager exposes a
/// simple state machine (`.idle → .loading → .ready`) and provides
/// `loadModel(modelId:)` for explicit loads and `unloadModel()` to release
/// the ~2 GB working set when the app is backgrounded.
///
/// ## Usage
/// ```swift
/// // Load a model (no-op if already loaded)
/// try await MLXModelManager.shared.loadModel(modelId: "mlx-community/Qwen2.5-3B-Instruct-4bit")
///
/// // Generate tokens
/// let container = await MLXModelManager.shared.getContainer()!
/// let stream = try await generateFromModel(container: container, messages: messages)
/// for await token in stream { /* append to UI */ }
/// ```
///
/// ## Memory management
/// The `com.apple.developer.kernel.increased-memory-limit` entitlement
/// (already set in Onyx.entitlements) allows the app to keep a 2 GB model
/// resident without being jetsam-killed on devices with 6+ GB RAM.
public actor MLXModelManager {

    /// Shared singleton — use this throughout the app.
    public static let shared = MLXModelManager()

    /// Lifecycle state of the model manager.
    ///
    /// Observe this from the UI with `.task { await model.state }` or by
    /// polling `ChatProvider.modelStatus`.
    public enum State: Sendable {
        /// No model loaded. Initial state after launch or after `unloadModel()`.
        case idle
        /// A model is being loaded from disk. UI should show a progress indicator.
        case loading
        /// Model is resident and ready for inference.
        case ready
        /// Load failed. The associated string is a user-readable description.
        case failed(String)
    }

    /// Current lifecycle state. Changes are observable via Swift concurrency.
    public private(set) var state: State = .idle

    /// HuggingFace id of the model currently in memory, or nil if none.
    ///
    /// This is set to the requested id before loading begins so the UI can
    /// show "Loading Qwen 2.5…" even while the weights are being read.
    public private(set) var currentModelId: String?

    private var container: ModelContainer?

    private init() {}

    // MARK: - Load / Unload

    /// Load a model from the local models store into memory.
    ///
    /// - If the requested model is already loaded, this is a **no-op**.
    /// - If a *different* model is loaded, it is unloaded first so the device
    ///   never holds two ~2 GB weight sets simultaneously.
    /// - Throws `MLXError.modelNotInstalled` if the directory is missing (the
    ///   user must download the model first via `ChatModelDownloader`).
    ///
    /// - Parameter modelId: HuggingFace model id, e.g.
    ///   `"mlx-community/Qwen2.5-3B-Instruct-4bit"`.
    public func loadModel(modelId: String) async throws {
        // Already loaded — nothing to do.
        if case .ready = state, currentModelId == modelId { return }
        // Same model already loading — don't start a duplicate load.
        if case .loading = state, currentModelId == modelId { return }

        // Hot-swap: unload old model first to avoid brief double-residency.
        if currentModelId != nil && currentModelId != modelId {
            unloadModel()
        }

        currentModelId = modelId
        state = .loading

        do {
            guard let modelDir = Self.findInstalledModel(modelId: modelId) else {
                let msg = "Model '\(modelId)' is not installed. Go to the Models tab to download it."
                state = .failed(msg)
                currentModelId = nil
                throw MLXError.modelNotInstalled(id: modelId)
            }

            let loaded = try await LLMModelFactory.shared.loadContainer(
                from: modelDir,
                using: HubTokenizerLoader()
            )

            container = loaded
            state = .ready
        } catch {
            state = .failed(error.localizedDescription)
            throw error
        }
    }

    /// Ensure a model is loaded. If it's already loading for the same id,
    /// this waits (polling at 200 ms intervals) until `.ready` or `.failed`.
    ///
    /// Use this from `ChatProvider` instead of `loadModel` to safely handle
    /// the race where two chat turns are triggered in quick succession.
    public func ensureLoaded(modelId: String) async throws {
        switch state {
        case .ready where currentModelId == modelId:
            return
        case .loading where currentModelId == modelId:
            // Poll until the in-flight load settles.
            while true {
                try await Task.sleep(for: .milliseconds(200))
                if case .ready = state, currentModelId == modelId { return }
                if case .failed(let msg) = state {
                    throw MLXError.modelLoadFailed(msg)
                }
            }
        default:
            try await loadModel(modelId: modelId)
        }
    }

    /// Unload the resident model and reclaim its memory.
    ///
    /// Call this when the app enters the background (via
    /// `scenePhase == .background`) to avoid jetsam on 6 GB devices.
    /// The next chat turn will lazy-reload automatically.
    public func unloadModel() {
        container = nil
        currentModelId = nil
        state = .idle
    }

    // MARK: - Container access

    /// Returns the loaded `ModelContainer`, or nil if no model is resident.
    ///
    /// Pass this to `generateFromModel(container:messages:)` to run inference.
    public func getContainer() -> ModelContainer? {
        container
    }

    // MARK: - Filesystem helpers

    /// Returns the on-disk URL for `modelId` if `config.json` exists there.
    ///
    /// This is the same existence check used by `ChatModelRegistry`. Exposed
    /// as a `nonisolated static` so it can be called from outside the actor
    /// without `await`.
    nonisolated public static func findInstalledModel(modelId: String) -> URL? {
        let dir = OnyxPaths.modelDirectory(for: modelId)
        let hasConfig = FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("config.json").path
        )
        return hasConfig ? dir : nil
    }
}

// MARK: - Token streaming

nonisolated func promptTokenIds(
    for messages: [[String: String]],
    tokenizer: any MLXLMCommon.Tokenizer,
    modelId: String?
) -> [Int] {
    do {
        return try tokenizer.applyChatTemplate(messages: messages)
    } catch MLXLMCommon.TokenizerError.missingChatTemplate {
        let prompt = fallbackChatPrompt(for: messages, modelId: modelId)
        return tokenizer.encode(text: prompt, addSpecialTokens: false)
    } catch {
        let prompt = fallbackChatPrompt(for: messages, modelId: modelId)
        return tokenizer.encode(text: prompt, addSpecialTokens: false)
    }
}

nonisolated public func fallbackChatPrompt(for messages: [[String: String]], modelId: String?) -> String {
    let family = modelId.flatMap { ChatModelCatalog.descriptor(forId: $0)?.family }

    switch family {
    case .qwen:
        return qwenFallbackPrompt(for: messages)
    case .gemma:
        return gemmaFallbackPrompt(for: messages)
    default:
        return simpleFallbackPrompt(for: messages)
    }
}
nonisolated private func qwenFallbackPrompt(for messages: [[String: String]]) -> String {
    var prompt = ""
    
    // Qwen natively supports a baseline system persona to steady its reasoning
    prompt += "<|im_start|>system\nYou are a helpful, fluid, and natural conversational assistant.<|im_end|>\n"
    
    for message in messages {
        let role = message["role"] == "assistant" ? "assistant" : "user"
        let content = message["content"] ?? ""
        prompt += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
    }
    
    // Force pre-fill generation cue for the assistant's streaming return
    prompt += "<|im_start|>assistant\n"
    return prompt
}
nonisolated private func gemmaFallbackPrompt(for messages: [[String: String]]) -> String {
    var prompt = ""
    for message in messages {
        let role = message["role"] == "assistant" ? "model" : "user"
        let content = message["content"] ?? ""
        prompt += "<start_of_turn>\(role)\n\(content)<end_of_turn>\n"
    }
    prompt += "<start_of_turn>model\n"
    return prompt
}

nonisolated private func simpleFallbackPrompt(for messages: [[String: String]]) -> String {
    var prompt = ""
    for message in messages {
        let role = message["role"] == "assistant" ? "Assistant" : "User"
        let content = message["content"] ?? ""
        prompt += "\(role):\n\(content)\n\n"
    }
    prompt += "Assistant:\n"
    return prompt
}

nonisolated func shouldBufferGeneratedText(modelId: String?) -> Bool {
    modelId.flatMap { ChatModelCatalog.descriptor(forId: $0)?.family } == .gemma
}

nonisolated func visibleGeneratedText(from rawText: String, modelId: String?) -> (text: String, shouldStop: Bool) {
    guard shouldBufferGeneratedText(modelId: modelId) else { return (rawText, false) }

    var text = rawText
    while let stripped = text.strippingLeadingGemmaRoleLine() {
        text = stripped
    }

    let stopMarkers = [
        "<end_of_turn>",
        "<start_of_turn>",
        "\nmodel\n",
        "\nmodel ",
        "\nuser\n",
        "\nuser ",
        "model\n",
        "user\n"
    ]

    var firstStopRange: Range<String.Index>?
    for marker in stopMarkers {
        guard let range = text.range(of: marker) else { continue }
        if firstStopRange == nil || range.lowerBound < firstStopRange!.lowerBound {
            firstStopRange = range
        }
    }

    guard let firstStopRange else { return (text, false) }
    return (String(text[..<firstStopRange.lowerBound]), true)
}

private extension String {
    func strippingLeadingGemmaRoleLine() -> String? {
        let trimmedPrefix = drop(while: { $0 == "\n" || $0 == " " || $0 == "\t" })
        let rolePrefixes = [
            "<start_of_turn>model\n",
            "<start_of_turn>model ",
            "model\n",
            "model "
        ]

        for prefix in rolePrefixes where trimmedPrefix.hasPrefix(prefix) {
            return String(trimmedPrefix.dropFirst(prefix.count))
        }

        return nil
    }
}

/// Stream tokens from a loaded model for a given conversation history.
///
/// This is the core inference function. It applies the model's native chat
/// template to `messages`, then streams tokens through `container.generate`.
///
/// - Parameters:
///   - container: A loaded `ModelContainer` from `MLXModelManager.getContainer()`.
///   - messages: Array of `["role": "user"|"assistant", "content": "..."]`
///     dictionaries. Build this with `MLXConversationHistory.buildMessages(systemPrompt:)`.
///   - maxTokens: Hard cap on output length. Default 1024.
///   - temperature: Sampling temperature. Higher = more creative. Default 0.6.
///   - topP: Nucleus sampling threshold. Default 0.9.
/// - Returns: An `AsyncStream<String>` that yields token chunks as they are
///   generated. Consume with `for await chunk in stream { … }`.
///
/// ## Example
/// ```swift
/// let stream = try await generateFromModel(
///     container: container,
///     messages: [["role": "user", "content": "Hello!"]]
/// )
/// var reply = ""
/// for await chunk in stream {
///     reply += chunk
///     label.text = reply   // update UI incrementally
/// }
/// ```
///
/// - Important: This function is `nonisolated` so it does not run on the
///   `@MainActor` even though the project-wide `SWIFT_DEFAULT_ACTOR_ISOLATION`
///   is set to `MainActor`. Inference is CPU/GPU-intensive and must not block
///   the main thread.
nonisolated public func generateFromModel(
    container: ModelContainer,
    messages: [[String: String]],
    modelId: String? = nil,
    maxKVSize: Int = 2048, //added
    maxTokens: Int = 384, //Int = 1024,
    temperature: Float = 0.7, //Float = 0.6,
    topP: Float = 0.85 //Float = 0.9
) async throws -> AsyncStream<String> {
    let tokenizer = await container.tokenizer
    let tokenIds = promptTokenIds(for: messages, tokenizer: tokenizer, modelId: modelId)
    let lmInput = LMInput(tokens: MLXArray(tokenIds))
    let params = GenerateParameters(maxTokens: maxTokens, maxKVSize: maxKVSize, temperature: temperature, topP: topP)

    let generationStream = try await container.generate(
        input: lmInput,
        parameters: params
    )

    return AsyncStream<String> { continuation in
        Task {
            let buffersUntilStop = shouldBufferGeneratedText(modelId: modelId)
            var rawText = ""
            var emittedText = ""

            // The labelled break is required: a plain `break` inside a
            // `switch` only exits the switch, not the enclosing for-await.
            tokenLoop: for await event in generationStream {
                switch event {
                case .chunk(let text):
                    rawText += text
                    let visible = visibleGeneratedText(from: rawText, modelId: modelId)

                    if !buffersUntilStop && visible.text.count > emittedText.count {
                        let delta = visible.text.dropFirst(emittedText.count)
                        continuation.yield(String(delta))
                        emittedText = visible.text
                    }
                    if visible.shouldStop { break tokenLoop }
                case .info:
                    break
                default:
                    break
                }
            }

            if buffersUntilStop {
                let visible = visibleGeneratedText(from: rawText, modelId: modelId)
                if !visible.text.isEmpty {
                    continuation.yield(visible.text)
                }
            }
            continuation.finish()
        }
    }
}
