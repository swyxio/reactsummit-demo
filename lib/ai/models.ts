export const DEFAULT_CHAT_MODEL: string = "chat-model";

export type ChatModel = {
  id: string;
  name: string;
  description: string;
};

export const chatModels: ChatModel[] = [
  {
    id: "chat-model",
    name: "ZAI GLM 4.6",
    description: "Fast and efficient language model via Cerebras",
  },
  {
    id: "chat-model-reasoning",
    name: "ZAI GLM 4.6 Reasoning",
    description:
      "Uses advanced chain-of-thought reasoning for complex problems",
  },
  {
    id: "chat-model-gpt5",
    name: "GPT-5",
    description: "OpenAI's latest flagship model with advanced capabilities",
  },
];
