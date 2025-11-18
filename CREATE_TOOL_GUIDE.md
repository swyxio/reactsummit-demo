# Create Tool Meta-Tool Guide

## Overview

The `createTool` meta-tool is a development-only feature that allows you to dynamically generate new AI tools with complete implementations, including:

- Tool implementation with Zod schema validation
- Custom UI component for displaying results
- Streaming progress updates during creation
- Integration instructions for manual setup

## Features

### 1. **Intelligent Tool Planning**
- Analyzes your description and creates a detailed implementation plan
- Determines input/output schemas automatically
- Suggests appropriate UI component types (card, table, chart, custom)

### 2. **Streaming Progress Updates**
Watch in real-time as the tool:
- Generates the tool implementation
- Creates the UI component
- Writes files to disk
- Provides integration instructions

### 3. **Complete Code Generation**
- Production-ready TypeScript code
- Proper error handling
- Type-safe schemas with Zod
- Modern React components with Tailwind CSS

## Usage

**In development mode only**, ask the AI to create a new tool:

```
"Create a tool that fetches stock prices for a given ticker symbol"
"Make a tool to generate random jokes"
"Build a tool that calculates mortgage payments"
```

## Example Flow

1. **User Request**: "Create a tool that gets cryptocurrency prices"

2. **Tool Execution** (with streaming progress):
   - ✓ Planning
   - ⏳ Generate tool implementation...
   - ✓ Generate UI component...
   - ✓ Write tool file → `lib/ai/tools/getCryptoPrices.ts`
   - ✓ Write component file → `components/getCryptoPrices.tsx`
   - ✓ Generate integration instructions

3. **Output**: Integration instructions with code snippets for:
   - Adding to chat route
   - Adding type definitions
   - Adding UI rendering in message component

## Files Created

### Tool Implementation (`lib/ai/tools/[toolName].ts`)
```typescript
import { tool } from "ai";
import { z } from "zod";

export const yourTool = tool({
  description: "...",
  inputSchema: z.object({
    // Generated schema
  }),
  execute: async (input) => {
    // Generated implementation
  },
});
```

### UI Component (`components/[toolName].tsx`)
```tsx
export default function YourToolComponent({ data }) {
  // Generated React component with Tailwind styling
}
```

## Integration Steps

After the tool generates files, you need to manually:

1. **Add to Chat Route** (`app/(chat)/api/chat/route.ts`):
   - Import the tool
   - Add to `experimental_activeTools` array
   - Add to `tools` object

2. **Add Type Definition** (`lib/types.ts`):
   - Import the tool type
   - Add to `ChatTools` type

3. **Add UI Rendering** (`components/message.tsx`):
   - Import the component
   - Add rendering logic in the `message.parts?.map()` loop

4. **Restart Dev Server**: Hot reload will pick up the new files

## Architecture

### Data Flow
```
User Request
    ↓
AI Model (createTool)
    ↓
Plan Generation (streamObject)
    ↓
Progress Updates (dataStream.write)
    ↓
Tool Implementation (streamObject)
    ↓
UI Component (streamObject)
    ↓
File Writing (fs.writeFile)
    ↓
Integration Instructions
```

### Key Components

- **`lib/ai/tools/create-tool.ts`**: Core meta-tool implementation
- **`components/tool-creation-progress.tsx`**: Progress UI components
- **`components/tool-creation-stream.tsx`**: Streaming progress handler
- **`lib/types.ts`**: Type definitions including `toolCreationProgress`

### Streaming Progress

Progress updates are sent via `dataStream.write()` with type `data-toolCreationProgress`:

```typescript
dataStream.write({
  type: "data-toolCreationProgress",
  data: {
    step: "Generate tool implementation",
    status: "in-progress",
    detail: "Creating getTool...",
  },
  transient: true,
});
```

The `ToolCreationStream` component listens to these updates and displays them in real-time.

## Limitations

### Development Only
The tool is **only available in development mode** for security reasons. It:
- Writes files to disk
- Executes code generation
- Should not be exposed in production

### Manual Integration Required
Generated files need manual integration because:
- Type safety requires explicit imports
- Hot reload needs proper module resolution
- Ensures developer review before deployment

### No Automatic Testing
Generated tools should be:
- Manually tested
- Reviewed for security
- Validated for functionality

## Best Practices

### 1. **Clear Descriptions**
Be specific about what the tool should do:
- ✅ "Create a tool that fetches weather data for a city using the OpenWeather API"
- ❌ "Make a weather thing"

### 2. **Review Generated Code**
Always review:
- Input validation
- Error handling
- API security (API keys, rate limits)
- Edge cases

### 3. **Test Thoroughly**
- Test with various inputs
- Verify error handling
- Check UI responsiveness
- Validate data accuracy

### 4. **Iterate**
If the generated tool doesn't meet your needs:
- Edit the files directly
- Use `updateDocument` tool for refinements
- Regenerate with more specific instructions

## Troubleshooting

### Files Not Found
- Check file paths in output
- Ensure proper permissions
- Verify cwd is project root

### Type Errors
- Restart TypeScript server
- Check import paths
- Verify type definitions added to `lib/types.ts`

### Hot Reload Not Working
- Restart dev server
- Clear Next.js cache (`.next` folder)
- Check for syntax errors

### Tool Not Appearing
- Verify `isDevelopmentEnvironment` is true
- Check tool is added to `experimental_activeTools`
- Ensure tool object is in `tools` config

## Example Generated Tools

### Stock Price Checker
```typescript
// lib/ai/tools/getStockPrice.ts
export const getStockPrice = tool({
  description: "Get current stock price for a ticker symbol",
  inputSchema: z.object({
    ticker: z.string(),
  }),
  execute: async ({ ticker }) => {
    // Implementation
  },
});
```

### Joke Generator
```typescript
// lib/ai/tools/generateJoke.ts
export const generateJoke = tool({
  description: "Generate a random joke",
  inputSchema: z.object({
    category: z.enum(["programming", "general", "dad"]),
  }),
  execute: async ({ category }) => {
    // Implementation
  },
});
```

## Future Enhancements

Potential improvements:
- Automatic integration (if safe)
- Test generation
- Documentation generation
- Version control integration
- Tool marketplace/sharing

## Security Considerations

- **Never** expose in production
- Review generated code for security issues
- Validate API keys are not hardcoded
- Check for injection vulnerabilities
- Sanitize user inputs

---

For more information on the generative UI architecture, see the main README.
