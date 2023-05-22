import { OpenAI } from "langchain/llms/openai";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import "dotenv/config";
import * as fs from "fs";
import * as readline from "readline-sync";

export const run = async () => {
  /* Initialize the LLM to use to answer the question */
  const model = new OpenAI({});
  /* Load in the file we want to do question answering over */
  console.log("Reading file")
  const text = fs.readFileSync("the_long_guide.txt", "utf8");
  /* Split the text into chunks */
  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
  const docs = await textSplitter.createDocuments([text]);
  /* Create the vectorstore */
  console.log("Vector store created\n\n")
  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
  /* Create the chain */
  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorStore.asRetriever()
  );
  /* Ask it a question */
  const question = readline.question("Ask a question about payday 2: ")
  const res = await chain.call({ question, chat_history: [] });
  console.log(res.text);
  /* Ask it a follow up question */
  while (true) {
    let chatHistory = question + res.text;
    const followUpQuestion = readline.question("Ask a follow up question: ");
    const followUpRes = await chain.call({
      question: followUpQuestion,
      chat_history: chatHistory,
    });
    console.log(followUpRes.text + "\n");
    chatHistory += followUpQuestion + followUpRes.text;
  }
};

run();