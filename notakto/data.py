#https://arxiv.org/pdf/1301.1672v1.pdf

data = {
    ((0,0,0),(0,0,0),(0,0,0)):"c",
    ((0,0,0),(0,1,0),(0,0,0)):"cc",
    ((1,1,0),(0,0,0),(0,0,0)):"ad",
    ((1,0,1),(0,0,0),(0,0,0)):"b",
    ((1,0,0),(0,1,0),(0,0,0)):"b",
    ((1,0,0),(0,0,1),(0,0,0)):"b",
    ((1,0,0),(0,0,0),(0,0,1)):"a",

    ((0,1,0),(1,0,0),(0,0,0)):"a",
    ((0,1,0),(0,1,0),(0,0,0)):"b",
    ((0,1,0),(0,0,0),(0,1,0)):"a",
    ((1,1,0),(1,0,0),(0,0,0)):"b",
    ((1,0,0),(1,1,0),(0,0,0)):"ab",
    ((1,1,0),(0,0,1),(0,0,0)):"d",
    ((1,1,0),(0,0,0),(1,0,0)):"a",
    ((1,1,0),(0,1,0),(0,0,0)):"d",

    ((1,1,0),(0,0,0),(0,0,1)):"d",
    ((1,0,1),(0,1,0),(0,0,0)):"a",
    ((1,0,1),(0,0,0),(1,0,0)):"ab",
    ((1,0,1),(0,0,0),(0,1,0)):"a",
    ((1,0,0),(0,1,1),(0,0,0)):"a",
    ((0,1,0),(1,1,0),(0,0,0)):"ab",
    ((0,1,0),(1,0,1),(0,0,0)):"b",

    ((1,1,0),(1,1,0),(0,0,0)):"a",
    ((1,1,0),(1,0,1),(0,0,0)):"a",
    ((1,1,0),(1,0,0),(0,0,1)):"a",
    ((1,1,0),(0,1,1),(0,0,0)):"b",

    ((1,1,0),(0,1,0),(1,0,0)):"b",
    ((1,1,0),(0,0,1),(1,0,0)):"b",
    ((1,1,0),(0,0,1),(0,1,0)):"ab",
    ((1,1,0),(0,0,1),(0,0,1)):"ab",
    ((1,1,0),(0,0,0),(1,1,0)):"b",
    ((1,1,0),(0,0,0),(1,0,1)):"b",
    ((1,1,0),(0,0,0),(0,1,1)):"a",

    ((1,0,1),(0,1,0),(0,1,0)):"b",
    ((1,0,1),(0,0,0),(1,0,1)):"a",
    ((1,0,0),(0,1,1),(0,1,0)):"b",
    ((0,1,0),(1,0,1),(0,1,0)):"a",

    ((1,1,0),(1,0,1),(0,1,0)):"b",

    ((1,1,0),(1,0,1),(0,0,1)):"b",
    ((1,1,0),(0,1,1),(1,0,0)):"a",
    ((1,1,0),(0,0,1),(1,1,0)):"a",
    ((1,1,0),(0,0,1),(1,0,1)):"a",

    ((1,1,0),(1,0,1),(0,1,1)):"a",
}

reductions = {"aa":"","bbb":"b","bbc":"c",
            "ccc":"acc","bbd":"d","cd":"ad",
            "dd":"cc","":"aa", "b":"bbb",
            "c":"bbc","acc":"ccc","d":"bbd",
            "ad":"cd","cc":"dd"
}

targets = "a bb bc cc".split()
