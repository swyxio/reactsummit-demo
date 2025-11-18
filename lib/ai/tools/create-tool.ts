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
      console.log(
        "[createTool] Starting tool creation with description:",
        description
      );

      // Only allow in development
      if (!isDevelopmentEnvironment) {
        console.error(
          "[createTool] Attempted tool creation outside development mode"
        );
        return {
          error: "Tool creation is only available in development mode",
        };
      }

      console.log("[createTool] Development mode confirmed, proceeding...");

      // Step 1: Generate plan
      console.log("[createTool] Step 1: Starting planning phase");
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
      console.log("[createTool] Requesting tool metadata from AI model");
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

      // Stream the plan as it's being generated, with early feedback
      let lastPlanLength = 0;
      let hasShownToolName = false;
      let hasShownInputSchema = false;
      let hasShownOutputSchema = false;
      const allPlanSteps: string[] = [];

      console.log("[createTool] Streaming tool metadata from AI...");
      for await (const partialObject of toolMetadataStream.partialObjectStream) {
        // Show tool name as soon as it's available
        if (partialObject.toolName && !hasShownToolName) {
          console.log(
            "[createTool] Tool name received:",
            partialObject.toolName
          );
          dataStream.write({
            type: "data-toolCreationProgress",
            data: {
              step: "Defining tool structure",
              status: "in-progress",
              detail: `Tool name: ${partialObject.toolName}`,
            },
            transient: true,
          });
          hasShownToolName = true;
        }

        // Show input schema progress
        if (partialObject.inputSchema?.fields && !hasShownInputSchema) {
          console.log(
            "[createTool] Input schema received:",
            partialObject.inputSchema.fields.length,
            "fields"
          );
          dataStream.write({
            type: "data-toolCreationProgress",
            data: {
              step: "Defining input parameters",
              status: "in-progress",
              detail: `${partialObject.inputSchema.fields.length} parameter(s) defined`,
            },
            transient: true,
          });
          hasShownInputSchema = true;
        }

        // Show output schema progress
        if (partialObject.outputSchema?.fields && !hasShownOutputSchema) {
          console.log(
            "[createTool] Output schema received:",
            partialObject.outputSchema.fields.length,
            "fields"
          );
          dataStream.write({
            type: "data-toolCreationProgress",
            data: {
              step: "Defining output structure",
              status: "in-progress",
              detail: `${partialObject.outputSchema.fields.length} field(s) defined`,
            },
            transient: true,
          });
          hasShownOutputSchema = true;
        }

        // Stream implementation plan steps as they're generated
        if (
          partialObject.implementationPlan &&
          partialObject.implementationPlan.length > lastPlanLength
        ) {
          const newSteps =
            partialObject.implementationPlan.slice(lastPlanLength);
          for (const planStep of newSteps) {
            if (planStep?.step) {
              console.log(
                "[createTool] Implementation plan step added:",
                planStep.step
              );
              allPlanSteps.push(planStep.step);
              dataStream.write({
                type: "data-toolCreationProgress",
                data: {
                  step: planStep.step,
                  status: "pending",
                  detail: planStep.description,
                },
                transient: true,
              });
            }
          }
          lastPlanLength = partialObject.implementationPlan.length;
        }
      }

      const toolMetadata = await toolMetadataStream.object;
      console.log("[createTool] Tool metadata complete:", {
        toolName: toolMetadata.toolName,
        displayName: toolMetadata.displayName,
        implementationPlanSteps: toolMetadata.implementationPlan.length,
      });

      // Mark intermediate steps as completed
      if (hasShownToolName) {
        dataStream.write({
          type: "data-toolCreationProgress",
          data: {
            step: "Defining tool structure",
            status: "completed",
            detail: `Tool: ${toolMetadata.displayName}`,
          },
          transient: true,
        });
      }

      if (hasShownInputSchema) {
        dataStream.write({
          type: "data-toolCreationProgress",
          data: {
            step: "Defining input parameters",
            status: "completed",
          },
          transient: true,
        });
      }

      if (hasShownOutputSchema) {
        dataStream.write({
          type: "data-toolCreationProgress",
          data: {
            step: "Defining output structure",
            status: "completed",
          },
          transient: true,
        });
      }

      // Mark planning as completed
      console.log("[createTool] Planning phase completed");
      dataStream.write({
        type: "data-toolCreationProgress",
        data: {
          step: "Planning",
          status: "completed",
          detail: `Ready to implement ${toolMetadata.displayName}`,
        },
        transient: true,
      });

      // Mark all implementation plan steps as completed since they're just planning steps
      console.log(
        "[createTool] Marking",
        allPlanSteps.length,
        "implementation plan steps as completed"
      );
      for (const planStep of allPlanSteps) {
        console.log("[createTool] Completing plan step:", planStep);
        dataStream.write({
          type: "data-toolCreationProgress",
          data: {
            step: planStep,
            status: "completed",
          },
          transient: true,
        });
      }

      // Add remaining execution steps to the plan
      console.log("[createTool] Setting up execution steps");
      const executionSteps = [
        { step: "Generate tool implementation", status: "pending" as const },
        { step: "Generate UI component", status: "pending" as const },
        { step: "Write tool file", status: "pending" as const },
        { step: "Write component file", status: "pending" as const },
        {
          step: "Generate integration instructions",
          status: "pending" as const,
        },
      ];

      // Send execution plan steps
      for (const item of executionSteps) {
        dataStream.write({
          type: "data-toolCreationProgress",
          data: { step: item.step, status: item.status },
          transient: true,
        });
      }

      // Step 2: Generate tool implementation
      console.log("[createTool] Step 2: Generating tool implementation");
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

      console.log(
        "[createTool] Waiting for tool implementation to complete..."
      );
      const toolImplementation = await toolImplementationStream.object;
      console.log("[createTool] Tool implementation received:", {
        codeLength: toolImplementation.code.length,
        importsCount: toolImplementation.imports.length,
      });

      dataStream.write({
        type: "data-toolCreationProgress",
        data: {
          step: "Generate tool implementation",
          status: "completed",
        },
        transient: true,
      });

      // Step 3: Generate UI component
      console.log("[createTool] Step 3: Generating UI component");
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

      console.log(
        "[createTool] Waiting for component implementation to complete..."
      );
      const componentImplementation =
        await componentImplementationStream.object;
      console.log("[createTool] Component implementation received:", {
        componentName: componentImplementation.componentName,
        codeLength: componentImplementation.code.length,
      });

      dataStream.write({
        type: "data-toolCreationProgress",
        data: {
          step: "Generate UI component",
          status: "completed",
        },
        transient: true,
      });

      // Step 4: Write files
      console.log("[createTool] Step 4: Writing tool file");
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

      console.log("[createTool] Writing tool file to:", toolFilePath);
      await writeFile(toolFilePath, toolImplementation.code, "utf-8");
      console.log("[createTool] Tool file written successfully");

      dataStream.write({
        type: "data-toolCreationProgress",
        data: {
          step: "Write tool file",
          status: "completed",
          detail: `lib/ai/tools/${toolMetadata.toolName}.ts`,
        },
        transient: true,
      });

      console.log("[createTool] Writing component file");
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

      console.log("[createTool] Writing component file to:", componentFilePath);
      await writeFile(componentFilePath, componentImplementation.code, "utf-8");
      console.log("[createTool] Component file written successfully");

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
      console.log("[createTool] Step 5: Generating integration instructions");
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

      const result = {
        toolName: toolMetadata.toolName,
        displayName: toolMetadata.displayName,
        filesCreated: [
          `lib/ai/tools/${toolMetadata.toolName}.ts`,
          `components/${toolMetadata.toolName}.tsx`,
        ],
        integrationInstructions,
        success: true,
      };

      console.log("[createTool] Tool creation completed successfully:", result);
      return result;
    },
  });
