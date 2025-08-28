/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
import { GoogleGenAI, Modality } from '@google/genai';
import { LanguageCode, languageNameMap } from '../utils/translations';

// --- Gemini Configuration ---
// The API key is hardcoded to allow the app to run outside of the development environment,
// such as on a Netlify deployment. This is necessary because client-side JavaScript
// does not have access to 'process.env'.
const ai = new GoogleGenAI({ apiKey: 'AIzaSyAWh63Lki_c2V8ox7UdsRz7xZ2Ow6XnjbE' });
// --- End Gemini Configuration ---

export type GenerationMode = 'encyclopedia' | 'eli5' | 'practicalExamples' | 'stepByStep' | 'summary' | 'funFacts';

type StreamedContent = {
  text?: string;
  imageUrl?: string;
  error?: string;
};

const getPromptForMode = (topic: string, language: LanguageCode, mode: GenerationMode): string => {
  const fullLanguageName = languageNameMap[language] || 'English';
  const commonInstructions = `The response must be in ${fullLanguageName}. Be informative. Do not use markdown, titles, or any special formatting. Respond with only the text of the response itself.`;
  const imageInstruction = `Also, generate a single illustrative, high-quality, landscape-format image relevant to the topic "${topic}".`;

  let textPrompt: string;

  switch (mode) {
    case 'eli5':
      textPrompt = `Explain the term "${topic}" in 2-3 simple sentences, with common words, as if you were explaining it to a 5-year-old child. ${commonInstructions}`;
      break;
    case 'practicalExamples':
      textPrompt = `Provide a concise explanation of "${topic}" (around 3-4 sentences) focused on its practical applications and real-world examples. ${commonInstructions}`;
      break;
    case 'stepByStep':
      textPrompt = `Explain how "${topic}" works or is done in a maximum of 5 clear, sequential steps. Start each step on a new line with a number (1., 2., 3., ...). ${commonInstructions}`;
      break;
    case 'summary':
      textPrompt = `Provide a schematic summary of the key points for "${topic}". Present it as a short, unordered list of 3-5 points. Start each point on a new line with a bullet point (·). ${commonInstructions}`;
      break;
    case 'funFacts':
      textPrompt = `Provide a short, unordered list of 3-5 interesting and little-known fun facts about "${topic}". Start each fact on a new line with a bullet point (·). ${commonInstructions}`;
      break;
    case 'encyclopedia':
    default:
      textPrompt = `Provide a concise (around 4-6 sentences), technical, precise, and complete encyclopedia-style single-paragraph definition for the term: "${topic}". Be neutral. ${commonInstructions}`;
      break;
  }
  
  return `${textPrompt} ${imageInstruction}`;
};

/**
 * Streams both a definition and an image for a given topic from the Gemini API in a single call.
 * @param topic The word or term to define.
 * @param language The language for the response.
 * @param mode The generation mode for the content.
 * @returns An async generator that yields objects containing text chunks, a final image URL, or an error.
 */
export async function* generateContentAndImageStream(
  topic: string,
  language: LanguageCode,
  mode: GenerationMode,
): AsyncGenerator<StreamedContent, void, undefined> {
  const prompt = getPromptForMode(topic, language, mode);

  try {
    const responseStream = await ai.models.generateContentStream({
       model: "gemini-2.5-flash-image-preview",
       contents: prompt,
       config: {
         responseModalities: [Modality.IMAGE, Modality.TEXT],
       },
    });
    
    for await (const chunk of responseStream) {
      // Yield text as it comes in
      if (chunk.text) {
        yield { text: chunk.text };
      }
      
      // Check for and yield the image part
      for (const part of chunk.candidates?.[0]?.content?.parts || []) {
        if (part.inlineData?.data && part.inlineData?.mimeType) {
          const base64ImageBytes: string = part.inlineData.data;
          const imageUrl = `data:${part.inlineData.mimeType};base64,${base64ImageBytes}`;
          yield { imageUrl };
        }
      }
    }
  } catch (error) {
    console.error('Error streaming from Gemini:', error);
    const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred.';
    yield { error: `Could not generate content for "${topic}". ${errorMessage}` };
  }
}
