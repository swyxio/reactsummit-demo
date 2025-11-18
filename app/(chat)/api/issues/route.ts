import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import matter from "gray-matter";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

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

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const filename = searchParams.get("filename");

    const issuesDir = join(process.cwd(), "issues");

    // If filename is provided, return just that issue
    if (filename) {
      const filePath = join(issuesDir, filename);
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

      return NextResponse.json({
        filename,
        metadata: data as IssueMetadata,
        content: filteredContent,
      });
    }

    // Otherwise, return paginated list
    const limit = Number.parseInt(searchParams.get("limit") || "20", 10);
    const offset = Number.parseInt(searchParams.get("offset") || "0", 10);
    const files = await readdir(issuesDir);

    // Filter for markdown files and sort by filename (reverse chronological)
    const mdFiles = files
      .filter((file) => file.endsWith(".md"))
      .sort()
      .reverse();

    // Parse frontmatter for the paginated subset
    const paginatedFiles = mdFiles.slice(offset, offset + limit);
    const issues: Issue[] = await Promise.all(
      paginatedFiles.map(async (issueFilename) => {
        const filePath = join(issuesDir, issueFilename);
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

        return {
          filename: issueFilename,
          metadata: data as IssueMetadata,
          content: filteredContent,
        };
      })
    );

    return NextResponse.json({
      issues,
      hasMore: offset + limit < mdFiles.length,
      total: mdFiles.length,
    });
  } catch (error) {
    console.error("Error fetching issues:", error);
    return NextResponse.json(
      { error: "Failed to fetch issues" },
      { status: 500 }
    );
  }
}
