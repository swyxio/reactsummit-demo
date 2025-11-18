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
      return;
    }

    const toolCreationMessages = dataStream.filter(
      (part) => part.type === "data-toolCreationProgress"
    );

    if (toolCreationMessages.length > 0) {
      setProgressUpdates((prev) => {
        const updated = [...prev];

        for (const msg of toolCreationMessages) {
          const existingIndex = updated.findIndex(
            (p) => p.step === msg.data.step
          );

          if (existingIndex >= 0) {
            // Update existing step
            updated[existingIndex] = msg.data;
          } else {
            // Add new step
            updated.push(msg.data);
          }
        }

        return updated;
      });
    }
  }, [dataStream]);

  if (progressUpdates.length === 0) {
    return null;
  }

  return (
    <div className="space-y-2">
      {progressUpdates.map((progress, index) => (
        <ToolCreationProgress
          key={`${progress.step}-${index}`}
          progress={progress}
        />
      ))}
    </div>
  );
}
