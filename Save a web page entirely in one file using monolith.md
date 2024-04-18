# Save a web page entirely in one file using monolith

I found a fantastic tool called [monolith](https://github.com/Y2Z/monolith) for storing entire web pages inside a single HTML file.
It does this by embedding assets as base64-encoded data, so the entire page can load without any network connection. Great for archival!

Example:

```shell
monolith 'https://lethain.com/mental-model-for-how-to-use-llms-in-products/' -I -o 'Notes on how to use LLMs in your product.html'
```
