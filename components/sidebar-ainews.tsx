"use client";

import { motion } from "framer-motion";
import { useRouter } from "next/navigation";
import useSWRInfinite from "swr/infinite";
import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "@/components/ui/sidebar";
import { fetcher } from "@/lib/utils";
import { LoaderIcon } from "./icons";

type IssueMetadata = {
  id: string;
  title: string;
  date: string;
  description: string;
  companies?: string[];
  models?: string[];
  topics?: string[];
  people?: string[];
};

type Issue = {
  filename: string;
  metadata: IssueMetadata;
  content: string;
};

type IssuesResponse = {
  issues: Issue[];
  hasMore: boolean;
  total: number;
};

const PAGE_SIZE = 20;
const DATE_REGEX = /^(\d{2}-\d{2}-\d{2})/;

function getIssuesPaginationKey(
  pageIndex: number,
  previousPageData: IssuesResponse
) {
  if (previousPageData && previousPageData.hasMore === false) {
    return null;
  }

  const offset = pageIndex * PAGE_SIZE;
  return `/api/issues?limit=${PAGE_SIZE}&offset=${offset}`;
}

export function SidebarAINews() {
  const { setOpenMobile } = useSidebar();
  const router = useRouter();

  const {
    data: paginatedIssues,
    setSize,
    isValidating,
    isLoading,
  } = useSWRInfinite<IssuesResponse>(getIssuesPaginationKey, fetcher, {
    fallbackData: [],
  });

  const hasReachedEnd = paginatedIssues
    ? paginatedIssues.some((page) => page.hasMore === false)
    : false;

  const hasEmptyIssues = paginatedIssues
    ? paginatedIssues.every((page) => page.issues.length === 0)
    : false;

  const handleIssueClick = (issue: Issue) => {
    // Just navigate to display the article, don't create a chat yet
    router.push(`/?article=${encodeURIComponent(issue.filename)}`);
    setOpenMobile(false);
  };

  if (isLoading) {
    return (
      <SidebarGroup>
        <SidebarGroupLabel>AI News</SidebarGroupLabel>
        <SidebarGroupContent>
          <div className="flex flex-col">
            {[44, 32, 28, 64, 52].map((item) => (
              <div
                className="flex h-8 items-center gap-2 rounded-md px-2"
                key={item}
              >
                <div
                  className="h-4 max-w-(--skeleton-width) flex-1 rounded-md bg-sidebar-accent-foreground/10"
                  style={
                    {
                      "--skeleton-width": `${item}%`,
                    } as React.CSSProperties
                  }
                />
              </div>
            ))}
          </div>
        </SidebarGroupContent>
      </SidebarGroup>
    );
  }

  if (hasEmptyIssues) {
    return (
      <SidebarGroup>
        <SidebarGroupLabel>AI News</SidebarGroupLabel>
        <SidebarGroupContent>
          <div className="flex w-full flex-row items-center justify-center gap-2 px-2 text-sm text-zinc-500">
            No AI news articles found.
          </div>
        </SidebarGroupContent>
      </SidebarGroup>
    );
  }

  return (
    <SidebarGroup>
      <SidebarGroupLabel>AI News</SidebarGroupLabel>
      <SidebarGroupContent>
        <SidebarMenu>
          {paginatedIssues?.flatMap((page) =>
            page.issues.map((issue) => {
              // Extract date from filename or use metadata date
              const dateMatch = issue.filename.match(DATE_REGEX);
              const displayDate = dateMatch ? dateMatch[1] : "";

              return (
                <SidebarMenuItem key={issue.filename}>
                  <SidebarMenuButton
                    className="data-[active=true]:bg-sidebar-accent data-[active=true]:text-sidebar-accent-foreground"
                    onClick={() => handleIssueClick(issue)}
                  >
                    <div className="flex flex-col gap-1 overflow-hidden">
                      <span className="line-clamp-1 text-sm">
                        {issue.metadata.title}
                      </span>
                      <span className="text-sidebar-foreground/50 text-xs">
                        {displayDate}
                      </span>
                    </div>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              );
            })
          )}
        </SidebarMenu>

        <motion.div
          onViewportEnter={() => {
            if (!isValidating && !hasReachedEnd) {
              setSize((size) => size + 1);
            }
          }}
        />

        {hasReachedEnd ? (
          <div className="mt-4 flex w-full flex-row items-center justify-center gap-2 px-2 text-sm text-zinc-500">
            End of AI News
          </div>
        ) : (
          <div className="mt-4 flex flex-row items-center gap-2 p-2 text-zinc-500 dark:text-zinc-400">
            <div className="animate-spin">
              <LoaderIcon />
            </div>
            <div>Loading News...</div>
          </div>
        )}
      </SidebarGroupContent>
    </SidebarGroup>
  );
}
