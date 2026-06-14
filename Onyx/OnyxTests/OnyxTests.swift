//
//  OnyxTests.swift
//  OnyxTests
//
//  Created by Errol Brandt on 6/6/2026.
//

import XCTest
@testable import Onyx

final class OnyxTests: XCTestCase {

    func testHardwareProfileUsesPreciseRAMForModelGate() throws {
        let profile = HardwareProfile(
            tier: .base,
            detectedPCores: 2,
            detectedRAMMegabytes: 7_800,
            detectedRAMGigabytes: 7,
            detectedGPUMemoryMB: 0,
            isEnvOverridden: false,
            chipBrand: "Test Chip"
        )

        let modelSizeBytes = Int64(5_752 * 1_048_576)

        XCTAssertTrue(profile.canLoadModel(approxSizeBytes: modelSizeBytes))
    }

    func testHardwareProfileRejectsWhenPreciseRAMIsInsufficient() throws {
        let profile = HardwareProfile(
            tier: .base,
            detectedPCores: 2,
            detectedRAMMegabytes: 7_168,
            detectedRAMGigabytes: 7,
            detectedGPUMemoryMB: 0,
            isEnvOverridden: false,
            chipBrand: "Test Chip"
        )

        let modelSizeBytes = Int64(5_752 * 1_048_576)

        XCTAssertFalse(profile.canLoadModel(approxSizeBytes: modelSizeBytes))
    }

    func testGemmaFallbackPromptUsesGemmaTurnMarkers() throws {
        let prompt = fallbackChatPrompt(
            for: [
                ["role": "user", "content": "Hello"],
                ["role": "assistant", "content": "Hi"],
                ["role": "user", "content": "What can you do?"]
            ],
            modelId: "mlx-community/gemma-4-e4b-it-4bit"
        )

        XCTAssertEqual(
            prompt,
            "<start_of_turn>user\nHello<end_of_turn>\n" +
            "<start_of_turn>model\nHi<end_of_turn>\n" +
            "<start_of_turn>user\nWhat can you do?<end_of_turn>\n" +
            "<start_of_turn>model\n"
        )
    }

    func testGemmaVisibleTextStopsBeforeRepeatedModelTurn() throws {
        let visible = visibleGeneratedText(
            from: "Hi! How can I assist you today?\nmodel \nmodel\nHi! How can I assist you today?",
            modelId: "mlx-community/gemma-4-e4b-it-4bit"
        )

        XCTAssertEqual(visible.text, "Hi! How can I assist you today?")
        XCTAssertTrue(visible.shouldStop)
    }

    func testGemmaVisibleTextStripsLeadingModelRoleEcho() throws {
        let visible = visibleGeneratedText(
            from: "model\nHi! How can I assist you today?",
            modelId: "mlx-community/gemma-4-e4b-it-4bit"
        )

        XCTAssertEqual(visible.text, "Hi! How can I assist you today?")
        XCTAssertFalse(visible.shouldStop)
    }

    func testGemmaVisibleTextStopsBeforeSerializedConversationEcho() throws {
        let visible = visibleGeneratedText(
            from: "Hello<end_of_turn>\n<start_of_turn>user\nWhat is the capital of Canada<end_of_turn>",
            modelId: "mlx-community/gemma-4-e4b-it-4bit"
        )

        XCTAssertEqual(visible.text, "Hello")
        XCTAssertTrue(visible.shouldStop)
    }

    func testGemmaVisibleTextStripsEchoedStartOfTurnModelPrefix() throws {
        let visible = visibleGeneratedText(
            from: "<start_of_turn>model\nHello<end_of_turn>",
            modelId: "mlx-community/gemma-4-e4b-it-4bit"
        )

        XCTAssertEqual(visible.text, "Hello")
        XCTAssertTrue(visible.shouldStop)
    }
}
