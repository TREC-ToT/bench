import json
import logging
from pathlib import Path
from typing import NamedTuple, Dict, List

import ir_datasets
from ir_datasets.formats import TrecQrels, BaseDocs, BaseQueries
from ir_datasets.indices import PickleLz4FullStore

NAME = "trec-tot"

log = logging.getLogger(__name__)


class TrecToTDoc(NamedTuple):
    title: str
    doc_id: int
    wikidata_id: str
    text: str
    sections: List[Dict[str, str]]


class TrecToTQuery(NamedTuple):
    query_id: str
    query: str


class TrecToTDocs(BaseDocs):

    def __init__(self, dlc):
        super().__init__()
        self._dlc = dlc

    def docs_iter(self):
        return iter(self.docs_store())

    def parse_sections(self, doc):
        sections = {}
        for s in doc["sections"]:
            sections[s["section"]] = doc["text"][s["start"]:s["end"]]
        doc["sections"] = sections
        return doc

    def _docs_iter(self):
        with self._dlc.stream() as stream:
            for line in stream:
                yield TrecToTDoc(**self.parse_sections(json.loads(line)))

    def docs_cls(self):
        return TrecToTDoc

    def docs_store(self, field='doc_id'):
        return PickleLz4FullStore(
            path=f'{ir_datasets.util.home_path()}/trec-tot24/docs.pklz4',
            init_iter_fn=self._docs_iter,
            data_cls=self.docs_cls(),
            lookup_field=field,
            index_fields=[field],
            count_hint=3185450
        )

    def docs_count(self):
        return self.docs_store().count()

    def docs_namespace(self):
        return f'{NAME}/{self._name}'

    def docs_lang(self):
        return 'en'


class LocalFileStream:
    def __init__(self, path):
        self._path = path

    def stream(self):
        return open(self._path, "rb")


class TrecToTQueries(BaseQueries):
    def __init__(self, name, dlc):
        super().__init__()
        self._name = name
        self._dlc = dlc

    def queries_iter(self):
        with self._dlc.stream() as stream:
            for line in stream:
                data = json.loads(line)
                yield TrecToTQuery(**data)

    def queries_cls(self):
        return TrecToTQuery

    def queries_namespace(self):
        return f'{NAME}/{self._name}'

    def queries_lang(self):
        return 'en'


def register(path):
    qrel_defs = {
        1: 'answer',
        0: 'not answer',
    }

    path = Path(path)

    # corpus
    corpus = path / "corpus.jsonl"

    for split in {"train-2024", "dev1-2024", "dev2-2024", "test-2024"}:
        name = split

        # queries
        queries = path / split / "queries.jsonl"
        if not queries.exists():
            log.warning(f"not loading '{split}' split: {queries} not found")
            continue

        components = [
            TrecToTDocs(LocalFileStream(corpus)),
            TrecToTQueries(name, LocalFileStream(queries)),
        ]

        has_qrel = False
        # no qrel for test set
        if split != "test-2024":
            qrel = path / split / "qrel.txt"
            has_qrel = True
            components.append(TrecQrels(LocalFileStream(qrel), qrel_defs))

        ds = ir_datasets.Dataset(
            *components
        )

        ir_datasets.registry.register(NAME + ":" + name, ds)
        log.info(f"registered: {NAME}:{name} [qrel={has_qrel}]")


if __name__ == '__main__':

    path = input("Enter data path:")

    register(path.strip())

    sets = []

    for split in {"train-2024", "dev1-2024", "dev2-2024", "test-2024"}:
        name = split
        sets.append(NAME + ":" + name)

    print(f"available sets: {sets}")
    q = None
    for name in sets:
        try:
            dataset = ir_datasets.load(name)
        except KeyError:
            print(f"error loading {name}, skipping!")
            continue

        n_q = 0
        for q in dataset.queries_iter():
            n_q += 1

        if "test" not in name:
            n_qrel = 0
            for qrel in dataset.qrels_iter():
                n_qrel += 1

            assert n_qrel == n_q

        print(name)
        print(f"n queries: {n_q}")
        print()

    print(f"example query: {q}")

    n_docs = 0
    dataset = ir_datasets.load("trec-tot:train-2024")
    doc = None
    for doc in dataset.docs_iter():
        n_docs += 1

    print(f"example doc: {doc}")

    print("corpus size: ", n_docs)
