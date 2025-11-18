import { writeFile } from "node:fs/promises";
import path from "node:path";
import { streamObject, tool, type UIMessageStreamWriter } from "ai";
import { z } from "zod";
import { isDevelopmentEnvironment } from "@/lib/constants";
import type { ChatMessage } from "@/lib/types";
import { myProvider } from "../providers";

type CreateToolProps = {
  dataStream: UIMessageStreamWriter<ChatMessage>;
};

export const createTool = ({ dataStream }: CreateToolProps) =>
  tool({
    description:
      "Create a new AI tool dynamically based on a description. This meta-tool generates the tool implementation, UI component, and integrates it into the chat system. DEV ONLY.",
    inputSchema: z.object({
      description: z
        .string()
        .describe(
          "A detailed description of what the tool should do (e.g., 'fetch stock prices', 'generate memes', 'calculate mortgage payments')"
        ),
    }),
    execute: async ({ description }) => {
      // Only allow in development
      if (!isDevelopmentEnvironment) {
        return {
          error: "Tool creation is only available in development mode",
        };
      }

      // Step 1: Generate plan
      dataStream.write({
        type: "data-toolCreationProgress",
        data: {
          step: "Planning",
          status: "in-progress",
          detail: "Analyzing requirements and creating implementation plan...",
        },
        transient: true,
      });

      // Generate tool metadata and implementation plan
      const toolMetadataStream = streamObject({
        model: myProvider.languageModel("artifact-model"),
        system: `You are an expert at designing AI tools. Given a description, create a detailed plan for implementing the tool.
        
The tool should:
1. Have a clear, concise name (camelCase, no spaces, descriptive)
2. Have a well-defined input schema using Zod
3. Have a clear description
4. Return structured data suitable for UI rendering

Consider:
- What external APIs might be needed?
- What input parameters are required?
- What data structure should be returned?
- What kind of UI component would best display the results?`,
        prompt: `Create a tool that: ${description}`,
        schema: z.object({
          toolName: z
            .string()
            .describe(
              "Tool name in camelCase (e.g., getStockPrice, generateMeme)"
            ),
          displayName: z
            .string()
            .describe("Human-readable name (e.g., Get Stock Price)"),
          toolDescription: z.string(),
          inputSchema: z.object({
            fields: z.array(
              z.object({
                name: z.string(),
                type: z.enum(["string", "number", "boolean", "array"]),
                description: z.string(),
                optional: z.boolean(),
              })
            ),
          }),
          outputSchema: z.object({
            fields: z.array(
              z.object({
                name: z.string(),
                type: z.string(),
                description: z.string(),
              })
            ),
          }),
          implementationPlan: z.array(
            z.object({
              step: z.string(),
              description: z.string(),
            })
          ),
          apiNeeded: z.boolean(),
          apiDescription: z.string().optional(),
          uiComponentType: z.enum(["card", "table", "chart", "custom"]),
        }),
      });

      const toolMetadata = await toolMetadataStream.object;

      const plan = [
        { step: "Generate tool implementation", status: "pending" as const },
        { step: "Generate UI component", status: "pending" as const },
        { step: "Write tool file", status: "pending" as const },
        { step: "Write component file", status: "pending" as const },
        {
          step: "Generate integration instructions",
          status: "pending" as const,
        },
      ];

      // Send initial plan
      for (const item of plan) {
        dataStream.write({
          type: "data-toolCreationProgress",
          data: { step: item.step, status: item.status },
          transient: true,
        });
      }

      dataStream.write({
        type: "data-toolCreationProgress",
        data: {
          step: "Planning",
          status: "completed",
          detail: `Tool: ${toolMetadata.displayName}`,
        },
        transient: true,
      });

      // Step 2: Generate tool implementation
      dataStream.write({
        type: "data-toolCreationProgress",
        data: {
          step: "Generate tool implementation",
          status: "in-progress",
          detail: `Creating ${toolMetadata.toolName}...`,
        },
        transient: true,
      });

      const toolImplementationStream = streamObject({
        model: myProvider.languageModel("artifact-model"),
        system: `You are an expert TypeScript developer. Generate a complete, production-ready AI tool implementation using the Vercel AI SDK.

The tool should:
1. Use the 'tool' function from 'ai' package
2. Use Zod for schema validation
3. Include proper TypeScript types
4. Handle errors gracefully
5. Return structured data
6. Include comments explaining the implementation

${toolMetadata.apiNeeded ? `The tool needs to call an external API: ${toolMetadata.apiDescription}` : "The tool should use mock/sample data for now."}`,
        prompt: `Create a tool implementation for: ${description}
        
Tool name: ${toolMetadata.toolName}
Description: ${toolMetadata.toolDescription}
Input fields: ${JSON.stringify(toolMetadata.inputSchema.fields)}
Output fields: ${JSON.stringify(toolMetadata.outputSchema.fields)}`,
        schema: z.object({
          code: z.string(),
          imports: z.array(z.string()),
          notes: z.string(),
        }),
      });

      const toolImplementation = await toolImplementationStream.object;

      dataStream.write({
        type: "data-toolCreationProgress",
        data: {
          step: "Generate tool implementation",
          status: "completed",
        },
        transient: true,
      });

      // Step 3: Generate UI component
      dataStream.write({
        type: "data-toolCreationProgress",
        data: {
          step: "Generate UI component",
          status: "in-progress",
          detail: `Creating ${toolMetadata.displayName} component...`,
        },
        transient: true,
      });

      const componentImplementationStream = streamObject({
        model: myProvider.languageModel("artifact-model"),
        system: `You are an expert React/TypeScript developer. Generate a beautiful, modern UI component to display the tool's output.

Requirements:
1. Use TypeScript with proper types
2. Use Tailwind CSS for styling (already configured)
3. Use shadcn/ui patterns (card, badge, etc.)
4. Make it responsive and accessible
5. Include proper loading/error states if needed
6. Export the component as default

Available utilities:
- Tailwind CSS classes
- cn() utility from "@/lib/utils" for conditional classes
- All standard React hooks`,
        prompt: `Create a React component to display output from: ${toolMetadata.displayName}

Component type: ${toolMetadata.uiComponentType}
Output data structure: ${JSON.stringify(toolMetadata.outputSchema.fields)}

The component should receive a prop like: ${toolMetadata.toolName}Data

Make it beautiful, modern, and user-friendly.`,
        schema: z.object({
          code: z.string(),
          componentName: z.string(),
          notes: z.string(),
        }),
      });

      const componentImplementation =
        await componentImplementationStream.object;

      dataStream.write({
        type: "data-toolCreationProgress",
        data: {
          step: "Generate UI component",
          status: "completed",
        },
        transient: true,
      });

      // Step 4: Write files
      dataStream.write({
        type: "data-toolCreationProgress",
        data: {
          step: "Write tool file",
          status: "in-progress",
        },
        transient: true,
      });

      const toolFilePath = path.join(
        process.cwd(),
        "lib",
        "ai",
        "tools",
        `${toolMetadata.toolName}.ts`
      );

      await writeFile(toolFilePath, toolImplementation.code, "utf-8");

      dataStream.write({
        type: "data-toolCreationProgress",
        data: {
          step: "Write tool file",
          status: "completed",
          detail: `lib/ai/tools/${toolMetadata.toolName}.ts`,
        },
        transient: true,
      });

      dataStream.write({
        type: "data-toolCreationProgress",
        data: {
          step: "Write component file",
          status: "in-progress",
        },
        transient: true,
      });

      const componentFilePath = path.join(
        process.cwd(),
        "components",
        `${toolMetadata.toolName}.tsx`
      );

      await writeFile(componentFilePath, componentImplementation.code, "utf-8");

      dataStream.write({
        type: "data-toolCreationProgress",
        data: {
          step: "Write component file",
          status: "completed",
          detail: `components/${toolMetadata.toolName}.tsx`,
        },
        transient: true,
      });

      // Step 5: Generate integration instructions
      dataStream.write({
        type: "data-toolCreationProgress",
        data: {
          step: "Generate integration instructions",
          status: "in-progress",
        },
        transient: true,
      });

      const integrationInstructions = `
# Integration Instructions for ${toolMetadata.displayName}

## Files Created:
- \`lib/ai/tools/${toolMetadata.toolName}.ts\` - Tool implementation
- \`components/${toolMetadata.toolName}.tsx\` - UI component

## Manual Steps Required:

### 1. Add tool to chat route
In \`app/(chat)/api/chat/route.ts\`:

\`\`\`typescript
// Add import at top
import { ${toolMetadata.toolName} } from "@/lib/ai/tools/${toolMetadata.toolName}";

// Add to experimental_activeTools array (around line 196)
experimental_activeTools: [
  "getWeather",
  "createDocument",
  "updateDocument",
  "requestSuggestions",
  "${toolMetadata.toolName}", // Add this
],

// Add to tools object (around line 202)
tools: {
  getWeather,
  createDocument: createDocument({ session, dataStream }),
  updateDocument: updateDocument({ session, dataStream }),
  requestSuggestions: requestSuggestions({ session, dataStream }),
  ${toolMetadata.toolName}, // Add this
},
\`\`\`

### 2. Add tool type definition
In \`lib/types.ts\`:

\`\`\`typescript
// Add import
import type { ${toolMetadata.toolName} } from "./ai/tools/${toolMetadata.toolName}";

// Add to ChatTools type
type ${toolMetadata.toolName}Tool = InferUITool<typeof ${toolMetadata.toolName}>;

export type ChatTools = {
  getWeather: weatherTool;
  createDocument: createDocumentTool;
  updateDocument: updateDocumentTool;
  requestSuggestions: requestSuggestionsTool;
  ${toolMetadata.toolName}: ${toolMetadata.toolName}Tool; // Add this
};
\`\`\`

### 3. Add UI rendering in message component
In \`components/message.tsx\` (around line 268, before "return null;"):

\`\`\`typescript
// Add import at top
import { ${componentImplementation.componentName} } from "./${toolMetadata.toolName}";

// Add this block in the message.parts?.map() loop
if (type === "tool-${toolMetadata.toolName}") {
  const { toolCallId, state } = part;

  return (
    <Tool defaultOpen={true} key={toolCallId}>
      <ToolHeader state={state} type="tool-${toolMetadata.toolName}" />
      <ToolContent>
        {state === "input-available" && (
          <ToolInput input={part.input} />
        )}
        {state === "output-available" && (
          <ToolOutput
            errorText={undefined}
            output={<${componentImplementation.componentName} ${toolMetadata.toolName}Data={part.output} />}
          />
        )}
      </ToolContent>
    </Tool>
  );
}
\`\`\`

## Notes:
${toolImplementation.notes}
${componentImplementation.notes}

## Hot Reload:
After completing these steps, the Next.js dev server should automatically detect the changes.
You may need to trigger a chat message to see the new tool in action.

## Testing:
Try asking the AI: "${description}"
`;

      dataStream.write({
        type: "data-toolCreationProgress",
        data: {
          step: "Generate integration instructions",
          status: "completed",
        },
        transient: true,
      });

      return {
        toolName: toolMetadata.toolName,
        displayName: toolMetadata.displayName,
        filesCreated: [
          `lib/ai/tools/${toolMetadata.toolName}.ts`,
          `components/${toolMetadata.toolName}.tsx`,
        ],
        integrationInstructions,
        success: true,
      };
    },
  });
