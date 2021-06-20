from pydantic import BaseModel


class Query(BaseModel):
    query: str


class QueryResult(Query):
    result: str


class QnARequest(Query):
    question: str


class QnAResponse(QnARequest):
    result: str
