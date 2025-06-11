# backend/app/services/ml/concept_extractor.py

import ast
import re
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import spacy
from loguru import logger
import asyncio

from ...core.config import settings
from ...utils.code_analyzer import CodeAnalyzer
from ...utils.metrics import track_metric

@dataclass
class ProgrammingConcept:
    """Represents a programming concept detected in student input"""
    name: str
    category: str  # syntax, data_structure, algorithm, pattern, error
    confidence: float
    context: str
    related_concepts: List[str]
    difficulty_level: float
    
    def dict(self):
        return {
            "name": self.name,
            "category": self.category,
            "confidence": self.confidence,
            "context": self.context,
            "related_concepts": self.related_concepts,
            "difficulty_level": self.difficulty_level
        }

class ConceptExtractor:
    """
    Extracts programming concepts from student messages and code
    """
    
    def __init__(self):
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize code analyzer
        self.code_analyzer = CodeAnalyzer()
        
        # Concept patterns and keywords
        self.concept_patterns = self._load_concept_patterns()
        
        # Concept relationships
        self.concept_graph = self._build_concept_graph()
        
        # Error pattern mapping
        self.error_patterns = self._load_error_patterns()
        
    def _load_concept_patterns(self) -> Dict[str, Dict]:
        """Load patterns for identifying programming concepts"""
        return {
            "loops": {
                "keywords": ["for", "while", "loop", "iterate", "iteration"],
                "code_patterns": [r"\bfor\s+\w+\s+in\b", r"\bwhile\s+.+:"],
                "error_keywords": ["infinite loop", "iteration", "range"],
                "difficulty": 0.4
            },
            "conditionals": {
                "keywords": ["if", "else", "elif", "condition", "branch"],
                "code_patterns": [r"\bif\s+.+:", r"\belse\s*:", r"\belif\s+.+:"],
                "error_keywords": ["condition", "boolean", "comparison"],
                "difficulty": 0.3
            },
            "functions": {
                "keywords": ["function", "def", "return", "parameter", "argument"],
                "code_patterns": [r"\bdef\s+\w+\s*\(", r"\breturn\s+"],
                "error_keywords": ["function", "call", "argument", "parameter"],
                "difficulty": 0.5
            },
            "recursion": {
                "keywords": ["recursion", "recursive", "base case", "call itself"],
                "code_patterns": [r"def\s+(\w+)\s*\([^)]*\):[^}]+\1\s*\("],
                "error_keywords": ["recursion", "stack overflow", "base case"],
                "difficulty": 0.8
            },
            "classes": {
                "keywords": ["class", "object", "method", "attribute", "instance"],
                "code_patterns": [r"\bclass\s+\w+", r"\bself\.", r"__init__"],
                "error_keywords": ["class", "object", "attribute", "method"],
                "difficulty": 0.6
            },
            "lists": {
                "keywords": ["list", "array", "append", "index", "slice"],
                "code_patterns": [r"\[.*\]", r"\.append\(", r"\[\d+\]"],
                "error_keywords": ["list", "index", "out of range"],
                "difficulty": 0.3
            },
            "dictionaries": {
                "keywords": ["dictionary", "dict", "key", "value", "mapping"],
                "code_patterns": [r"\{.*:.*\}", r"\.get\(", r"\[[\'\"].*[\'\"]\]"],
                "error_keywords": ["key error", "dictionary", "mapping"],
                "difficulty": 0.4
            },
            "exceptions": {
                "keywords": ["try", "except", "exception", "error", "catch"],
                "code_patterns": [r"\btry\s*:", r"\bexcept\s+\w+"],
                "error_keywords": ["exception", "error", "traceback"],
                "difficulty": 0.5
            },
            "imports": {
                "keywords": ["import", "module", "package", "library"],
                "code_patterns": [r"\bimport\s+\w+", r"\bfrom\s+\w+\s+import"],
                "error_keywords": ["import error", "module not found"],
                "difficulty": 0.2
            },
            "decorators": {
                "keywords": ["decorator", "@", "wrapper", "annotation"],
                "code_patterns": [r"@\w+", r"def\s+\w+\(func\)"],
                "error_keywords": ["decorator", "wrapper"],
                "difficulty": 0.8
            },
            "generators": {
                "keywords": ["generator", "yield", "iterator", "lazy"],
                "code_patterns": [r"\byield\s+", r"\.\_\_next\_\_"],
                "error_keywords": ["generator", "yield", "StopIteration"],
                "difficulty": 0.9
            },
            "async": {
                "keywords": ["async", "await", "coroutine", "asyncio"],
                "code_patterns": [r"\basync\s+def", r"\bawait\s+"],
                "error_keywords": ["async", "await", "coroutine"],
                "difficulty": 0.9
            }
        }
    
    def _build_concept_graph(self) -> Dict[str, List[str]]:
        """Build relationships between concepts"""
        return {
            "variables": ["data_types", "assignment", "scope"],
            "loops": ["iteration", "conditionals", "lists", "range"],
            "functions": ["parameters", "return", "scope", "recursion"],
            "recursion": ["functions", "base_case", "stack"],
            "classes": ["objects", "methods", "inheritance", "attributes"],
            "inheritance": ["classes", "polymorphism", "super"],
            "exceptions": ["error_handling", "try_except", "debugging"],
            "lists": ["indexing", "slicing", "iteration", "methods"],
            "dictionaries": ["keys", "values", "hashing", "methods"],
            "generators": ["iteration", "yield", "memory_efficiency"],
            "decorators": ["functions", "wrappers", "higher_order"],
            "async": ["concurrency", "coroutines", "event_loop"]
        }
    
    def _load_error_patterns(self) -> Dict[str, Dict]:
        """Load common error patterns and their associated concepts"""
        return {
            "SyntaxError": {
                "concepts": ["syntax", "indentation", "parsing"],
                "patterns": [r"invalid syntax", r"unexpected indent"]
            },
            "NameError": {
                "concepts": ["variables", "scope", "definition"],
                "patterns": [r"name '.*' is not defined"]
            },
            "TypeError": {
                "concepts": ["data_types", "operations", "functions"],
                "patterns": [r"unsupported operand type", r"object is not callable"]
            },
            "IndexError": {
                "concepts": ["lists", "indexing", "bounds"],
                "patterns": [r"list index out of range"]
            },
            "KeyError": {
                "concepts": ["dictionaries", "keys", "access"],
                "patterns": [r"KeyError:"]
            },
            "AttributeError": {
                "concepts": ["objects", "attributes", "methods"],
                "patterns": [r"has no attribute"]
            },
            "RecursionError": {
                "concepts": ["recursion", "stack", "base_case"],
                "patterns": [r"maximum recursion depth exceeded"]
            }
        }
    
    async def extract(
        self,
        message: str,
        code_snippet: Optional[str] = None
    ) -> List[ProgrammingConcept]:
        """
        Extract programming concepts from student input
        """
        start_time = datetime.utcnow()
        concepts = []
        
        try:
            # Extract from natural language
            nl_concepts = await self._extract_from_text(message)
            concepts.extend(nl_concepts)
            
            # Extract from code if provided
            if code_snippet:
                code_concepts = await self._extract_from_code(code_snippet)
                concepts.extend(code_concepts)
            
            # Extract from error messages
            error_concepts = self._extract_from_errors(message)
            concepts.extend(error_concepts)
            
            # Deduplicate and merge concepts
            concepts = self._merge_concepts(concepts)
            
            # Add related concepts
            concepts = self._add_related_concepts(concepts)
            
            # Sort by confidence
            concepts.sort(key=lambda c: c.confidence, reverse=True)
            
            # Log metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            await track_metric("concept_extraction_time", processing_time)
            await track_metric("concepts_extracted", len(concepts))
            
            logger.debug(f"🧩 Extracted {len(concepts)} concepts in {processing_time:.3f}s")
            
            return concepts[:10]  # Return top 10 concepts
            
        except Exception as e:
            logger.error(f"❌ Error extracting concepts: {str(e)}")
            return []
    
    async def _extract_from_text(self, text: str) -> List[ProgrammingConcept]:
        """Extract concepts from natural language"""
        concepts = []
        text_lower = text.lower()
        
        # Process with spaCy
        doc = self.nlp(text)
        
        for concept_name, pattern_info in self.concept_patterns.items():
            confidence = 0.0
            context_snippets = []
            
            # Check keywords
            for keyword in pattern_info["keywords"]:
                if keyword in text_lower:
                    confidence += 0.3
                    # Find context around keyword
                    for sent in doc.sents:
                        if keyword in sent.text.lower():
                            context_snippets.append(sent.text)
            
            # Check error keywords
            for error_keyword in pattern_info.get("error_keywords", []):
                if error_keyword in text_lower:
                    confidence += 0.2
            
            if confidence > 0:
                concepts.append(ProgrammingConcept(
                    name=concept_name,
                    category="natural_language",
                    confidence=min(confidence, 1.0),
                    context=" ".join(context_snippets[:2]),
                    related_concepts=self.concept_graph.get(concept_name, []),
                    difficulty_level=pattern_info["difficulty"]
                ))
        
        return concepts
    
    async def _extract_from_code(self, code: str) -> List[ProgrammingConcept]:
        """Extract concepts from code snippets"""
        concepts = []
        
        # Analyze code structure
        try:
            # Parse AST
            tree = ast.parse(code)
            ast_concepts = await self._analyze_ast(tree)
            concepts.extend(ast_concepts)
        except SyntaxError as e:
            # Code has syntax errors
            concepts.append(ProgrammingConcept(
                name="syntax_error",
                category="error",
                confidence=1.0,
                context=str(e),
                related_concepts=["syntax", "debugging"],
                difficulty_level=0.2
            ))
        
        # Pattern matching
        for concept_name, pattern_info in self.concept_patterns.items():
            for pattern in pattern_info.get("code_patterns", []):
                if re.search(pattern, code, re.MULTILINE):
                    concepts.append(ProgrammingConcept(
                        name=concept_name,
                        category="code_pattern",
                        confidence=0.8,
                        context=self._get_pattern_context(code, pattern),
                        related_concepts=self.concept_graph.get(concept_name, []),
                        difficulty_level=pattern_info["difficulty"]
                    ))
        
        return concepts
    
    async def _analyze_ast(self, tree: ast.AST) -> List[ProgrammingConcept]:
        """Analyze AST to extract concepts"""
        concepts = []
        
        class ConceptVisitor(ast.NodeVisitor):
            def __init__(self):
                self.found_concepts = []
            
            def visit_FunctionDef(self, node):
                self.found_concepts.append(("functions", node.name))
                # Check for recursion
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if hasattr(child.func, 'id') and child.func.id == node.name:
                            self.found_concepts.append(("recursion", node.name))
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                self.found_concepts.append(("classes", node.name))
                self.generic_visit(node)
            
            def visit_For(self, node):
                self.found_concepts.append(("loops", "for loop"))
                self.generic_visit(node)
            
            def visit_While(self, node):
                self.found_concepts.append(("loops", "while loop"))
                self.generic_visit(node)
            
            def visit_If(self, node):
                self.found_concepts.append(("conditionals", "if statement"))
                self.generic_visit(node)
            
            def visit_Try(self, node):
                self.found_concepts.append(("exceptions", "try-except"))
                self.generic_visit(node)
            
            def visit_List(self, node):
                self.found_concepts.append(("lists", "list literal"))
                self.generic_visit(node)
            
            def visit_Dict(self, node):
                self.found_concepts.append(("dictionaries", "dict literal"))
                self.generic_visit(node)
        
        visitor = ConceptVisitor()
        visitor.visit(tree)
        
        # Convert to ProgrammingConcept objects
        for concept_name, context in visitor.found_concepts:
            if concept_name in self.concept_patterns:
                concepts.append(ProgrammingConcept(
                    name=concept_name,
                    category="ast_analysis",
                    confidence=0.9,
                    context=f"Found in: {context}",
                    related_concepts=self.concept_graph.get(concept_name, []),
                    difficulty_level=self.concept_patterns[concept_name]["difficulty"]
                ))
        
        return concepts
    
    def _extract_from_errors(self, text: str) -> List[ProgrammingConcept]:
        """Extract concepts from error messages"""
        concepts = []
        
        for error_type, error_info in self.error_patterns.items():
            for pattern in error_info["patterns"]:
                if re.search(pattern, text, re.IGNORECASE):
                    for concept in error_info["concepts"]:
                        concepts.append(ProgrammingConcept(
                            name=concept,
                            category="error_related",
                            confidence=0.7,
                            context=f"Related to {error_type}",
                            related_concepts=self.concept_graph.get(concept, []),
                            difficulty_level=self.concept_patterns.get(
                                concept, {}
                            ).get("difficulty", 0.5)
                        ))
        
        return concepts
    
    def _get_pattern_context(self, code: str, pattern: str) -> str:
        """Get context around a pattern match"""
        match = re.search(pattern, code, re.MULTILINE)
        if match:
            start = max(0, match.start() - 50)
            end = min(len(code), match.end() + 50)
            return code[start:end].strip()
        return ""
    
    def _merge_concepts(self, concepts: List[ProgrammingConcept]) -> List[ProgrammingConcept]:
        """Merge duplicate concepts"""
        merged = {}
        
        for concept in concepts:
            if concept.name in merged:
                # Merge with existing
                existing = merged[concept.name]
                existing.confidence = max(existing.confidence, concept.confidence)
                if concept.context and concept.context not in existing.context:
                    existing.context += " | " + concept.context
            else:
                merged[concept.name] = concept
        
        return list(merged.values())
    
    def _add_related_concepts(self, concepts: List[ProgrammingConcept]) -> List[ProgrammingConcept]:
        """Add weakly related concepts based on concept graph"""
        all_concepts = {c.name: c for c in concepts}
        
        for concept in list(concepts):  # Use list() to avoid modifying while iterating
            for related in concept.related_concepts:
                if related not in all_concepts and related in self.concept_patterns:
                    # Add with lower confidence
                    all_concepts[related] = ProgrammingConcept(
                        name=related,
                        category="related",
                        confidence=concept.confidence * 0.3,
                        context=f"Related to {concept.name}",
                        related_concepts=self.concept_graph.get(related, []),
                        difficulty_level=self.concept_patterns[related]["difficulty"]
                    )
        
        return list(all_concepts.values())