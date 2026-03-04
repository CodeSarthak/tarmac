const CHARS_PER_TOKEN = 4;
const COUNT_TOKENS_TIMEOUT = 3000; // 3s timeout

interface CountTokensResponse {
  input_tokens: number;
}

export async function countTokens(text: string): Promise<number> {
  const apiKey = process.env.ANTHROPIC_API_KEY;

  if (!apiKey) {
    return heuristicCount(text);
  }

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), COUNT_TOKENS_TIMEOUT);

    const response = await fetch(
      "https://api.anthropic.com/v1/messages/count_tokens",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": apiKey,
          "anthropic-version": "2023-06-01",
        },
        body: JSON.stringify({
          model: "claude-sonnet-4-6-20250514",
          messages: [{ role: "user", content: text }],
        }),
        signal: controller.signal,
      }
    );

    clearTimeout(timeout);

    if (!response.ok) {
      return heuristicCount(text);
    }

    const data = (await response.json()) as CountTokensResponse;
    return data.input_tokens;
  } catch {
    return heuristicCount(text);
  }
}

export function heuristicCount(text: string): number {
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}
