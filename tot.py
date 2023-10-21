import json
from typing import NamedTuple, Dict, List
from pathlib import Path
import ir_datasets
from ir_datasets.formats import TrecQrels, BaseDocs, BaseQueries
from ir_datasets.indices import PickleLz4FullStore
import logging

NAME = "trec-tot"

log = logging.getLogger(__name__)


class TrecToTDoc(NamedTuple):
    page_title: str
    doc_id: int
    page_source: str
    wikidata_id: str
    text: str
    sections: Dict[str, str]
    infoboxes: List[Dict[str, str]]
    wikidata_classes: List[List[str]]


class TrecToTQuery(NamedTuple):
    query_id: str
    text: str
    title: str
    domain: str
    sentence_annotations: List[Dict]


class TrecToTDocs(BaseDocs):

    def __init__(self, dlc):
        super().__init__()
        self._dlc = dlc

    def docs_iter(self):
        return iter(self.docs_store())

    def _docs_iter(self):
        with self._dlc.stream() as stream:
            for line in stream:
                data = json.loads(line)
                yield TrecToTDoc(**data)

    def docs_cls(self):
        return TrecToTDoc

    def docs_store(self, field='doc_id'):
        return PickleLz4FullStore(
            path=f'{ir_datasets.util.home_path() / NAME}/docs.pklz4',
            init_iter_fn=self._docs_iter,
            data_cls=self.docs_cls(),
            lookup_field=field,
            index_fields=[field],
            count_hint=231852
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

                yield TrecToTQuery(query_id=data["id"],
                                   text=data["text"],
                                   title=data["title"],
                                   domain=data["domain"],
                                   sentence_annotations=data["sentence_annotations"])

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

    for split in {"train", "dev", "test"}:

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

        # no qrel for test set
        if split != "test":
            qrel = path / split / "qrel.txt"

            components.append(TrecQrels(LocalFileStream(qrel), qrel_defs))

        ds = ir_datasets.Dataset(
            *components
        )

        ir_datasets.registry.register(NAME + ":" + name, ds)
        log.info(f"registered: {NAME}:{name}")


if __name__ == '__main__':

    path = "./data"

    register(path.strip())

    sets = []

    for split in {"train", "dev"}:
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
    dataset = ir_datasets.load("trec-tot:train")
    doc = None
    for doc in dataset.docs_iter():
        n_docs += 1

    print(f"example doc: {doc}")

    print("corpus size: ", n_docs)
