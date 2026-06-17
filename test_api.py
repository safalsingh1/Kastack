import urllib.request, json

def ask(q):
    data = json.dumps({"query": q}).encode()
    req = urllib.request.Request(
        "http://localhost:5000/api/chat",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    r = urllib.request.urlopen(req, timeout=30)
    res = json.loads(r.read())
    answer = res["answer"]
    sources = list(res.get("sources", {}).keys())
    print(f"Q: {q}")
    print(f"A: {answer[:400]}")
    print(f"Sources: {sources}")
    print()

ask("What hobbies do people talk about in these conversations?")
ask("Do they talk about music or bands?")
ask("Tell me about the conversations in \"Topic 2\". Keywords: doing · impala · favorite · being · enjoy")
ask("Tell me about the conversations in \"Topic 3\". Keywords: dog · day · pets · cat · best")
