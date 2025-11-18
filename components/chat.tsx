"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import useSWR, { useSWRConfig } from "swr";
import { unstable_serialize } from "swr/infinite";
import { ChatHeader } from "@/components/chat-header";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { useArtifactSelector } from "@/hooks/use-artifact";
import { useAutoResume } from "@/hooks/use-auto-resume";
import { useChatVisibility } from "@/hooks/use-chat-visibility";
import type { Vote } from "@/lib/db/schema";
import { ChatSDKError } from "@/lib/errors";
import type { Attachment, ChatMessage } from "@/lib/types";
import type { AppUsage } from "@/lib/usage";
import { fetcher, fetchWithErrorHandlers, generateUUID } from "@/lib/utils";
import { Artifact } from "./artifact";
import { useDataStream } from "./data-stream-provider";
import { Response } from "./elements/response";
import { Messages } from "./messages";
import { MultimodalInput } from "./multimodal-input";
import { getChatHistoryPaginationKey } from "./sidebar-history";
import { toast } from "./toast";
import type { VisibilityType } from "./visibility-selector";

// Regex patterns for markdown header extraction
const HEADER_PATTERN = /^#{1,2}\s+/;
const HEADER_LEVEL_PATTERN = /^(#{1,2})/;
const HEADER_STRIP_PATTERN = /^#{1,2}\s+/;
const MARKDOWN_LINK_PATTERN = /\[([^\]]+)\]\([^)]+\)/g;

export function Chat({
  id,
  initialMessages,
  initialChatModel,
  initialVisibilityType,
  isReadonly,
  autoResume,
  initialLastContext,
  articleContext,
}: {
  id: string;
  initialMessages: ChatMessage[];
  initialChatModel: string;
  initialVisibilityType: VisibilityType;
  isReadonly: boolean;
  autoResume: boolean;
  initialLastContext?: AppUsage;
  articleContext?: string;
}) {
  const { visibilityType } = useChatVisibility({
    chatId: id,
    initialVisibilityType,
  });

  const { mutate } = useSWRConfig();
  const { setDataStream } = useDataStream();

  const [input, setInput] = useState<string>("");
  const [usage, setUsage] = useState<AppUsage | undefined>(initialLastContext);
  const [showCreditCardAlert, setShowCreditCardAlert] = useState(false);
  const [currentModelId, setCurrentModelId] = useState(initialChatModel);
  const currentModelIdRef = useRef(currentModelId);
  const [pendingArticleContext, setPendingArticleContext] = useState<
    string | undefined
  >(articleContext);

  useEffect(() => {
    currentModelIdRef.current = currentModelId;
  }, [currentModelId]);

  const {
    messages,
    setMessages,
    sendMessage,
    status,
    stop,
    regenerate,
    resumeStream,
  } = useChat<ChatMessage>({
    id,
    messages: initialMessages,
    experimental_throttle: 100,
    generateId: generateUUID,
    transport: new DefaultChatTransport({
      api: "/api/chat",
      fetch: fetchWithErrorHandlers,
      prepareSendMessagesRequest(request) {
        return {
          body: {
            id: request.id,
            message: request.messages.at(-1),
            selectedChatModel: currentModelIdRef.current,
            selectedVisibilityType: visibilityType,
            ...request.body,
          },
        };
      },
    }),
    onData: (dataPart) => {
      setDataStream((ds) => (ds ? [...ds, dataPart] : []));
      if (dataPart.type === "data-usage") {
        setUsage(dataPart.data);
      }
    },
    onFinish: () => {
      mutate(unstable_serialize(getChatHistoryPaginationKey));
    },
    onError: (error) => {
      if (error instanceof ChatSDKError) {
        // Check if it's a credit card error
        if (
          error.message?.includes("AI Gateway requires a valid credit card")
        ) {
          setShowCreditCardAlert(true);
        } else {
          toast({
            type: "error",
            description: error.message,
          });
        }
      }
    },
  });

  const searchParams = useSearchParams();
  const query = searchParams.get("query");

  const [hasAppendedQuery, setHasAppendedQuery] = useState(false);

  useEffect(() => {
    if (query && !hasAppendedQuery) {
      sendMessage({
        role: "user" as const,
        parts: [{ type: "text", text: query }],
      });

      setHasAppendedQuery(true);
      window.history.replaceState({}, "", `/chat/${id}`);
    }
  }, [query, sendMessage, hasAppendedQuery, id]);

  // Wrap sendMessage to prepend article context on first message
  const handleSendMessage: typeof sendMessage = (message, options) => {
    console.log("handleSendMessage called", {
      hasPendingContext: !!pendingArticleContext,
      messagesLength: messages.length,
      message,
    });

    if (
      pendingArticleContext &&
      messages.length === 0 &&
      message &&
      typeof message !== "string"
    ) {
      // This is the first message and we have article context
      let textContent = "";

      // Extract text from message
      if ("parts" in message && message.parts) {
        textContent = message.parts
          .filter(
            (p): p is Extract<typeof p, { type: "text" }> => p.type === "text"
          )
          .map((p) => p.text)
          .join("");
      } else if ("text" in message && typeof message.text === "string") {
        textContent = message.text;
      }

      console.log("Enhancing message with article context", { textContent });

      const enhancedMessage = {
        role: "user" as const,
        parts: [
          {
            type: "text" as const,
            text: `Here is an AI News article for context:\n\n${pendingArticleContext}\n\n---\n\nUser question: ${textContent}`,
          },
        ],
      };
      setPendingArticleContext(undefined); // Clear it after first use
      return sendMessage(enhancedMessage as typeof message, options);
    }
    return sendMessage(message, options);
  };

  const { data: votes } = useSWR<Vote[]>(
    messages.length >= 2 ? `/api/vote?chatId=${id}` : null,
    fetcher
  );

  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const isArtifactVisible = useArtifactSelector((state) => state.isVisible);
  const [isContentsExpanded, setIsContentsExpanded] = useState(true);
  const articleContentRef = useRef<HTMLDivElement>(null);

  useAutoResume({
    autoResume,
    initialMessages,
    resumeStream,
    setMessages,
  });

  return (
    <>
      <div className="overscroll-behavior-contain flex h-dvh min-w-0 touch-pan-y flex-col bg-background">
        <ChatHeader
          chatId={id}
          isReadonly={isReadonly}
          selectedVisibilityType={initialVisibilityType}
        />

        {pendingArticleContext && messages.length === 0 && (
          <div className="mx-auto w-full max-w-7xl px-2 pt-4 md:px-4">
            <div className="group relative rounded-lg border border-border bg-muted/30 shadow-sm">
              <div className="sticky top-0 z-10 flex items-center justify-between gap-2 border-border border-b bg-background/95 px-4 py-3 backdrop-blur supports-backdrop-filter:bg-background/60">
                <div className="flex items-center gap-2">
                  <span className="text-lg">📰</span>
                  <span className="font-semibold text-sm">Article Context</span>
                </div>
                <span className="text-muted-foreground text-xs">
                  Will be included with your first question
                </span>
              </div>

              {/* Table of Contents */}
              {(() => {
                const lines = pendingArticleContext.split("\n");
                const headers = lines
                  .filter((line) => line.match(HEADER_PATTERN))
                  .map((line) => {
                    const level =
                      line.match(HEADER_LEVEL_PATTERN)?.[0].length || 1;
                    let text = line.replace(HEADER_STRIP_PATTERN, "").trim();
                    // Strip markdown links - replace [text](url) with just text
                    text = text.replace(MARKDOWN_LINK_PATTERN, "$1");
                    // Generate anchor ID from text
                    const anchorId = text
                      .toLowerCase()
                      .replace(/[^\w\s-]/g, "")
                      .replace(/\s+/g, "-");
                    return { level, text, anchorId };
                  });

                if (headers.length > 0) {
                  return (
                    <div className="border-border border-b bg-muted/20">
                      <button
                        className="flex w-full items-center justify-between px-4 py-3 transition-colors hover:bg-muted/30"
                        onClick={() =>
                          setIsContentsExpanded(!isContentsExpanded)
                        }
                        type="button"
                      >
                        <div className="font-medium text-muted-foreground text-xs">
                          Contents
                        </div>
                        <svg
                          className={`h-4 w-4 transition-transform ${isContentsExpanded ? "rotate-180" : ""}`}
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            d="M19 9l-7 7-7-7"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                          />
                        </svg>
                      </button>
                      {isContentsExpanded && (
                        <div className="flex flex-wrap gap-x-4 gap-y-1 px-4 pb-3 text-sm">
                          {headers.map((header) => (
                            <button
                              className={`${
                                header.level === 1
                                  ? "font-semibold text-foreground"
                                  : "ml-4 text-muted-foreground"
                              } cursor-pointer text-left hover:underline`}
                              key={`${header.level}-${header.text}`}
                              onClick={() => {
                                const element =
                                  articleContentRef.current?.querySelector(
                                    `#${header.anchorId}`
                                  );
                                element?.scrollIntoView({
                                  behavior: "smooth",
                                  block: "start",
                                });
                              }}
                              type="button"
                            >
                              {header.text}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                }
                return null;
              })()}

              <div
                className="max-h-[50vh] overflow-y-auto p-4"
                ref={articleContentRef}
              >
                <Response className="prose prose-sm dark:prose-invert max-w-none [&_h1]:scroll-mt-4 [&_h2]:scroll-mt-4">
                  {pendingArticleContext
                    .split("\n")
                    .map((line) => {
                      if (line.match(HEADER_PATTERN)) {
                        let text = line
                          .replace(HEADER_STRIP_PATTERN, "")
                          .trim();
                        text = text.replace(MARKDOWN_LINK_PATTERN, "$1");
                        const anchorId = text
                          .toLowerCase()
                          .replace(/[^\w\s-]/g, "")
                          .replace(/\s+/g, "-");
                        return `<span id="${anchorId}"></span>\n${line}`;
                      }
                      return line;
                    })
                    .join("\n")}
                </Response>
              </div>
            </div>
          </div>
        )}

        <Messages
          chatId={id}
          isArtifactVisible={isArtifactVisible}
          isReadonly={isReadonly}
          messages={messages}
          regenerate={regenerate}
          selectedModelId={initialChatModel}
          setMessages={setMessages}
          status={status}
          votes={votes}
        />

        <div className="sticky bottom-0 z-1 mx-auto flex w-full max-w-4xl gap-2 border-t-0 bg-background px-2 pb-3 md:px-4 md:pb-4">
          {!isReadonly && (
            <MultimodalInput
              attachments={attachments}
              chatId={id}
              input={input}
              messages={messages}
              onModelChange={setCurrentModelId}
              selectedModelId={currentModelId}
              selectedVisibilityType={visibilityType}
              sendMessage={handleSendMessage}
              setAttachments={setAttachments}
              setInput={setInput}
              setMessages={setMessages}
              status={status}
              stop={stop}
              usage={usage}
            />
          )}
        </div>
      </div>

      <Artifact
        attachments={attachments}
        chatId={id}
        input={input}
        isReadonly={isReadonly}
        messages={messages}
        regenerate={regenerate}
        selectedModelId={currentModelId}
        selectedVisibilityType={visibilityType}
        sendMessage={handleSendMessage}
        setAttachments={setAttachments}
        setInput={setInput}
        setMessages={setMessages}
        status={status}
        stop={stop}
        votes={votes}
      />

      <AlertDialog
        onOpenChange={setShowCreditCardAlert}
        open={showCreditCardAlert}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Activate AI Gateway</AlertDialogTitle>
            <AlertDialogDescription>
              This application requires{" "}
              {process.env.NODE_ENV === "production" ? "the owner" : "you"} to
              activate Vercel AI Gateway.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                window.open(
                  "https://vercel.com/d?to=%2F%5Bteam%5D%2F%7E%2Fai%3Fmodal%3Dadd-credit-card",
                  "_blank"
                );
                window.location.href = "/";
              }}
            >
              Activate
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
