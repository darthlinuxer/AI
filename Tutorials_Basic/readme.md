# Basic Tutorials

At the end of Lesson 6, you will have a full working RAG to talk with:
1. Youtube link
2. Local PDF files
3. Online PDF files
4. Word files
5. Any website

```
Note: 
- All Basic Tutorials from lesson 1 to 5 have the Jupyter extension, which means they contain text and executable code in the Jupyter cells. Lesson 6 is not a Jupyter file but a regular python program
- execute lesson 6 by calling python3 '6.Rag v1 Example.py' from the Linux command line
```

## Expected output:

> `python3 '6.Rag v1 Example.py'`

**outputs**:
```
0. Exit
1. load from a new source
2. split into chunks semantically
3. Split into chunks with similar token count
4. add chunks to chroma collection
5. Talk to your collection
6. clean chroma collection
7. print loaded metadata
8. print loaded content
9. print loaded chunks
10. clear all variables
11. Send content to AI and get response
12. Print content token count
13. Print chunks token count
14. Add/update metadata
15. Add/Update content
16. get all chunks from vectordb using metadatas
17. get all chunks from vectordb using similarity with score
18. get all chunks from vectordb 

Enter your choice: 
```

### Normal workflow
1. load a document from a source (option 1)
2. split the loaded document content if it has more than 4096 tokens (use function 12 to check the size)
3. update the content metadata if needed (option 14)
4. add the splitted parts to local chroma db (option 4)
5. Talk to your document (option 5)