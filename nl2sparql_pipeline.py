"""
NL to SPARQL pipeline for media knowledge base exploration.
Converts natural language questions into SPARQL queries via entity linking.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Entity:
    surface: str
    uri: str
    entity_type: str
    confidence: float


@dataclass
class SPARQLResult:
    nl_query: str
    sparql: str
    entities: List[Entity]
    results: List[Dict]
    execution_time_ms: float


class EntityLinker:
    """
    Lightweight entity linker using a pre-built surface form dictionary.
    Maps surface mentions to knowledge base URIs (media entities).
    """

    def __init__(self, entity_dict: Optional[Dict[str, str]] = None):
        # entity_dict: {surface_form: uri}
        self.entity_dict = entity_dict or {}
        # Type hints for known URI prefixes
        self.type_map = {
            "ex:movie/": "Movie",
            "ex:person/": "Person",
            "ex:genre/": "Genre",
            "ex:show/": "TVShow",
        }

    def link(self, text: str) -> List[Entity]:
        entities = []
        text_lower = text.lower()
        for surface, uri in self.entity_dict.items():
            if surface.lower() in text_lower:
                etype = "Unknown"
                for prefix, t in self.type_map.items():
                    if uri.startswith(prefix):
                        etype = t
                        break
                entities.append(
                    Entity(
                        surface=surface,
                        uri=uri,
                        entity_type=etype,
                        confidence=0.92,
                    )
                )
        return entities


class QueryTemplateEngine:
    """
    Rule-based SPARQL template filler.
    Detects query intent and slots in linked entities.
    """

    TEMPLATES = {
        "genre": "SELECT ?title WHERE {{ ?m a ex:Movie ; ex:hasGenre {genre} ; ex:title ?title . }}",
        "director": "SELECT ?title WHERE {{ ?m a ex:Movie ; ex:directedBy {person} ; ex:title ?title . }}",
        "actor": "SELECT ?title WHERE {{ ?m a ex:Movie ; ex:hasActor {person} ; ex:title ?title . }}",
        "year": "SELECT ?title WHERE {{ ?m a ex:Movie ; ex:releaseYear {year} ; ex:title ?title . }}",
        "description": "SELECT ?desc WHERE {{ {entity} ex:description ?desc . }}",
        "fallback": "SELECT ?s ?p ?o WHERE {{ ?s ?p ?o . FILTER(CONTAINS(STR(?o), '{keyword}')) }} LIMIT 20",
    }

    def _detect_intent(self, text: str) -> str:
        t = text.lower()
        if any(w in t for w in ["direct", "directed by", "director"]):
            return "director"
        if any(w in t for w in ["act", "starring", "actor", "actress"]):
            return "actor"
        if any(w in t for w in ["genre", "action", "comedy", "drama", "thriller"]):
            return "genre"
        if re.search(r"\b(19|20)\d{2}\b", text):
            return "year"
        if any(w in t for w in ["about", "describe", "what is", "tell me"]):
            return "description"
        return "fallback"

    def fill(self, nl_query: str, entities: List[Entity]) -> str:
        intent = self._detect_intent(nl_query)
        entity_uri = entities[0].uri if entities else "?entity"
        year_match = re.search(r"\b(19|20)\d{2}\b", nl_query)
        year = year_match.group() if year_match else "?year"
        keyword = nl_query.split()[0] if nl_query else "media"

        slots = {
            "person": entity_uri,
            "genre": entity_uri,
            "entity": entity_uri,
            "year": year,
            "keyword": keyword,
        }
        template = self.TEMPLATES.get(intent, self.TEMPLATES["fallback"])
        try:
            return template.format(**slots)
        except KeyError:
            return self.TEMPLATES["fallback"].format(keyword=keyword)


class NL2SPARQLPipeline:
    def __init__(self, entity_dict: Optional[Dict[str, str]] = None):
        self.linker = EntityLinker(entity_dict)
        self.template_engine = QueryTemplateEngine()

    def translate(self, nl_query: str) -> SPARQLResult:
        import time

        start = time.perf_counter()
        entities = self.linker.link(nl_query)
        sparql = self.template_engine.fill(nl_query, entities)
        elapsed = (time.perf_counter() - start) * 1000

        return SPARQLResult(
            nl_query=nl_query,
            sparql=sparql,
            entities=entities,
            results=[],  # execution against actual triplestore not included
            execution_time_ms=round(elapsed, 2),
        )
