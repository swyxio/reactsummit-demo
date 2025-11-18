"use client";

import { useEffect, useState } from "react";
import { useDataStream } from "./data-stream-provider";
import { ToolCreationProgress } from "./tool-creation-progress";

export function ToolCreationStream() {
  const { dataStream } = useDataStream();
  const [progressUpdates, setProgressUpdates] = useState<
    Array<{
      step: string;
      status: "pending" | "in-progress" | "completed" | "error";
      detail?: string;
    }>
  >([]);

  useEffect(() => {
    if (!dataStream?.length) {
      console.log("[ToolCreationStream] No dataStream available");
      return;
    }

    const toolCreationMessages = dataStream.filter(
      (part) => part.type === "data-toolCreationProgress"
    );

    console.log(
      "[ToolCreationStream] Processing",
      toolCreationMessages.length,
      "tool creation messages"
    );

    if (toolCreationMessages.length > 0) {
      setProgressUpdates((prev) => {
        const updated = [...prev];

        for (const msg of toolCreationMessages) {
          console.log("[ToolCreationStream] Processing message:", msg.data);
          const existingIndex = updated.findIndex(
            (p) => p.step === msg.data.step
          );

          if (existingIndex >= 0) {
            // Update existing step
            console.log(
              "[ToolCreationStream] Updating existing step:",
              msg.data.step,
              "from",
              updated[existingIndex].status,
              "to",
              msg.data.status
            );
            updated[existingIndex] = msg.data;
          } else {
            // Add new step
            console.log(
              "[ToolCreationStream] Adding new step:",
              msg.data.step,
              "with status",
              msg.data.status
            );
            updated.push(msg.data);
          }
        }

        console.log(
          "[ToolCreationStream] Total progress steps:",
          updated.length
        );
        return updated;
      });
    }
  }, [dataStream]);

  if (progressUpdates.length === 0) {
    console.log("[ToolCreationStream] No progress updates to display");
    return null;
  }

  console.log(
    "[ToolCreationStream] Rendering",
    progressUpdates.length,
    "progress items"
  );

  const handleRetry = (step: string) => {
    console.log("[ToolCreationStream] Retry requested for step:", step);
    console.log(
      "[ToolCreationStream] Note: Retry functionality requires backend implementation"
    );
    // For now, just log - actual retry would need backend support
    alert(
      `Retry functionality for "${step}" requires backend implementation. Check console for details.`
    );
  };

  return (
    <div className="space-y-2">
      <div className="rounded-lg border bg-muted/20 p-3">
        <div className="font-medium text-sm">Progress:</div>
      </div>
      {progressUpdates.map((progress, index) => (
        <ToolCreationProgress
          key={`${progress.step}-${index}`}
          onRetry={handleRetry}
          progress={progress}
        />
      ))}
    </div>
  );
}
