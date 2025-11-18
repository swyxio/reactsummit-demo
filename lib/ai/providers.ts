import { cerebras } from "@ai-sdk/cerebras";
import { openai } from "@ai-sdk/openai";
import {
  customProvider,
  extractReasoningMiddleware,
  wrapLanguageModel,
} from "ai";
import { isTestEnvironment } from "../constants";

export const myProvider = isTestEnvironment
  ? (() => {
      const {
        artifactModel,
        chatModel,
        reasoningModel,
        titleModel,
        gpt5Model,
      } = require("./models.mock");
      return customProvider({
        languageModels: {
          "chat-model": chatModel,
          "chat-model-reasoning": reasoningModel,
          "chat-model-gpt5": gpt5Model,
          "title-model": titleModel,
          "artifact-model": artifactModel,
        },
      });
    })()
  : customProvider({
      languageModels: {
        "chat-model": cerebras("zai-glm-4.6"),
        "chat-model-reasoning": wrapLanguageModel({
          model: cerebras("zai-glm-4.6"),
          middleware: extractReasoningMiddleware({ tagName: "think" }),
        }),
        "chat-model-gpt5": openai("gpt-5"),
        "title-model": cerebras("zai-glm-4.6"),
        "artifact-model": cerebras("zai-glm-4.6"),
      },
    });
