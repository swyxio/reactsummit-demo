"use client";

import {
  CheckCircleIcon,
  CircleIcon,
  ClockIcon,
  CodeIcon,
  FileTextIcon,
  WrenchIcon,
  XCircleIcon,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

type ToolCreationProgressProps = {
  progress: {
    step: string;
    status: "pending" | "in-progress" | "completed" | "error";
    detail?: string;
  };
};

const getIcon = (step: string) => {
  if (step === "Planning") {
    return <WrenchIcon className="size-4" />;
  }
  if (step.includes("tool implementation")) {
    return <CodeIcon className="size-4" />;
  }
  if (step.includes("UI component")) {
    return <FileTextIcon className="size-4" />;
  }
  if (step.includes("file")) {
    return <FileTextIcon className="size-4" />;
  }
  if (step.includes("integration")) {
    return <CheckCircleIcon className="size-4" />;
  }
  return <CircleIcon className="size-4" />;
};

const getStatusIcon = (status: string) => {
  switch (status) {
    case "pending": {
      return <CircleIcon className="size-4 text-muted-foreground" />;
    }
    case "in-progress": {
      return <ClockIcon className="size-4 animate-pulse text-blue-600" />;
    }
    case "completed": {
      return <CheckCircleIcon className="size-4 text-green-600" />;
    }
    case "error": {
      return <XCircleIcon className="size-4 text-red-600" />;
    }
    default: {
      return <CircleIcon className="size-4" />;
    }
  }
};

const getStatusBadge = (status: string) => {
  const labels = {
    pending: "Pending",
    "in-progress": "In Progress",
    completed: "Completed",
    error: "Error",
  } as const;

  return (
    <Badge
      className={cn("text-xs", {
        "bg-gray-100 text-gray-700": status === "pending",
        "bg-blue-100 text-blue-700": status === "in-progress",
        "bg-green-100 text-green-700": status === "completed",
        "bg-red-100 text-red-700": status === "error",
      })}
      variant="secondary"
    >
      {getStatusIcon(status)}
      <span className="ml-1">{labels[status as keyof typeof labels]}</span>
    </Badge>
  );
};

export function ToolCreationProgress({ progress }: ToolCreationProgressProps) {
  return (
    <div className="my-2 rounded-lg border bg-muted/50 p-4">
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 flex-1 items-center gap-3">
          <div className="text-muted-foreground">{getIcon(progress.step)}</div>
          <div className="min-w-0 flex-1">
            <div className="font-medium text-sm">{progress.step}</div>
            {progress.detail && (
              <div className="mt-1 truncate text-muted-foreground text-xs">
                {progress.detail}
              </div>
            )}
          </div>
        </div>
        <div className="shrink-0">{getStatusBadge(progress.status)}</div>
      </div>
    </div>
  );
}

export function ToolCreationResult({ result }: { result: any }) {
  if (result.error) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-red-700 dark:bg-red-950/50">
        <div className="font-semibold">Error Creating Tool</div>
        <div className="mt-1 text-sm">{result.error}</div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border bg-background p-4">
      <div className="flex items-start gap-3">
        <div className="rounded-full bg-green-100 p-2 dark:bg-green-900/30">
          <CheckCircleIcon className="size-5 text-green-600" />
        </div>
        <div className="flex-1">
          <div className="font-semibold text-lg">{result.displayName}</div>
          <div className="mt-1 text-muted-foreground text-sm">
            Tool created successfully! Files generated:
          </div>
          <div className="mt-2 space-y-1">
            {result.filesCreated?.map((file: string) => (
              <div
                className="flex items-center gap-2 rounded bg-muted px-2 py-1 font-mono text-xs"
                key={file}
              >
                <FileTextIcon className="size-3" />
                {file}
              </div>
            ))}
          </div>
          {result.integrationInstructions && (
            <details className="mt-4">
              <summary className="cursor-pointer font-medium text-sm hover:underline">
                View Integration Instructions
              </summary>
              <pre className="mt-2 overflow-x-auto rounded bg-muted p-3 text-xs">
                {result.integrationInstructions}
              </pre>
            </details>
          )}
        </div>
      </div>
    </div>
  );
}
