# Chatbot API

A Chatbot API that relies on fine-tuned LLM for supporting multi-dialog RAG for constructing Family Audit plans.

AIxPA

- ``kind``: product
- ``ai``: NLP
- ``domain``: PA 

The chatbot exposes API that, given the documents as input, responds to the questions in order to construct suitable plan
for Family Audit actions. Specifically, it suggest the actions using the proposed documents as a baseline calling an approrpiately 
fine-tuned LLM API.

The products contains the operation for exposing the API.


## Usage

Tool usage documentation [here](./docs/usage.md).

## How To

- [Deploy, expose, and test the chatbot API in production mode](./docs/howto/deploy.md)
- [Deploy, expose, the chatbot API in mock mode (without LLM)](./docs/howto/deploymock.md)


## License

[Apache License 2.0](./LICENSE)
