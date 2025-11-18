import { readFile } from "node:fs/promises";
import { join } from "node:path";
import matter from "gray-matter";
import type { SearchParams } from "next/dist/server/request/search-params";
import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { Chat } from "@/components/chat";
import { DataStreamHandler } from "@/components/data-stream-handler";
import { DEFAULT_CHAT_MODEL } from "@/lib/ai/models";
import { generateUUID } from "@/lib/utils";
import { auth } from "../(auth)/auth";

export default async function Page(props: {
  searchParams: Promise<SearchParams>;
}) {
  const session = await auth();

  if (!session) {
    redirect("/api/auth/guest");
  }

  const id = generateUUID();
  const searchParams = await props.searchParams;
  const articleFilename = searchParams.article as string | undefined;

  // Load article metadata if article parameter is present (for display only)
  let articleContent: string | undefined;
  if (articleFilename) {
    try {
      console.log("Loading article from filesystem:", articleFilename);
      const issuesDir = join(process.cwd(), "issues");
      const filePath = join(issuesDir, articleFilename);
      const fileContent = await readFile(filePath, "utf-8");
      const { data, content } = matter(fileContent);

      // Filter out content below "Discord: Detailed by-Channel summaries and links"
      const discordSectionIndex = content.indexOf(
        "# Discord: Detailed by-Channel summaries and links"
      );
      const filteredContent =
        discordSectionIndex !== -1
          ? content.substring(0, discordSectionIndex).trim()
          : content;

      console.log("Article loaded:", data.title);
      // Format the article content for context
      articleContent = `# ${data.title}\n\n${data.description}\n\n---\n\n${filteredContent}`;
      console.log("Article content length:", articleContent.length);
    } catch (error) {
      console.error("Error loading article:", error);
    }
  }

  const cookieStore = await cookies();
  const modelIdFromCookie = cookieStore.get("chat-model");

  if (!modelIdFromCookie) {
    return (
      <>
        <Chat
          articleContext={articleContent}
          autoResume={false}
          id={id}
          initialChatModel={DEFAULT_CHAT_MODEL}
          initialMessages={[]}
          initialVisibilityType="private"
          isReadonly={false}
          key={`${id}-${articleFilename || "default"}`}
        />
        <DataStreamHandler />
      </>
    );
  }

  return (
    <>
      <Chat
        articleContext={articleContent}
        autoResume={false}
        id={id}
        initialChatModel={modelIdFromCookie.value}
        initialMessages={[]}
        initialVisibilityType="private"
        isReadonly={false}
        key={`${id}-${articleFilename || "default"}`}
      />
      <DataStreamHandler />
    </>
  );
}
