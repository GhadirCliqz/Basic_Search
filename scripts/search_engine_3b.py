from argparse import ArgumentParser
from itertools import chain, groupby
import json
from operator import itemgetter
import sys

from pyparsing import nestedExpr
from stemming.porter2 import stem
import math

class Posting(object):
    """
    Represents an element in the posting lists. It has the following fields:
    - doc_id: the document id: the original title.
    - tf: the term frequency in the document.
    """
    def __init__(self, doc_id=None, frequency=0):
        self.doc_id = doc_id
        self.tf = frequency

    def __repr__(self):
        return u"{}: {}".format(self.doc_id, self.tf).encode('utf-8')

    def __eq__(self, other):
        return self.doc_id == other.doc_id

class IndexStatistics(object):
    """
    A container for index statistics. It has the following fields:
    - no_docs: the number of documents in the index
    - no_unique_terms: the number of unique terms in the index (i.e. the number
                       of keys in the index)
    - no_terms: the total number of words in the index (which is the number of
                words in the corpus after filtering -- see the frequency field
                in the postings)
    - t_d_matrix: the size of the term-document matrix, if we actually created
                  one
    - index_size: the total length of all posting lists (the number of realised
                  "cells" from the term-document matrix)
    - saved_space: what percent (%) of the space required for the term-document
                   matrix have we saved by using an inverted index instead?
    """
    def __init__(self, index, corpus):
        # TODO
        self.no_docs = len(corpus)
        self.no_unique_terms = len(index)
        self.no_terms = self.count_terms(index)
        self.t_d_matrix = self.no_unique_terms * self.no_docs
        self.index_size = sum(len(v) for v in index.itervalues())
        self.saved_space = (1 - (float(self.index_size)/ self.t_d_matrix))*100
   
    def count_terms(self, index):
	s = 0
	for v in index.itervalues():
		s = s + sum(c.tf for c in v)
	return s


    def __repr__(self):
        return """Index statistics:
        Number of documents: {}
        Number of terms:     {}
        Unique terms:        {}
        Index size:          {}
	Term_doc matrix:          {}
	Saved space:          {}	
        """.format(self.no_docs, self.no_terms, self.no_unique_terms,
                   self.index_size, self.t_d_matrix, self.saved_space)

class QueryResponse(object):
    """
    Contains the documents returned for a query, as well as some statistics.
    """
    def __init__(self, results, docs_to_return=10):
        self.results = results[:docs_to_return]
        self.no_hits = len(results)

    def __repr__(self):
        ret = "Documents found: {}".format(self.no_hits)
        if len(self.results) > 0:
            for result in self.results:
                ret += "\n  {}: {}".format(*result)
        return ret

class SearchEngine(object):
    def __init__(self):
        self.index = None
        self.index_statistics = None
        self.stop_words = set()

    def load_file(self, corpus_file):
        """Loads and indexes the corpus file."""
        sys.stderr.write("Loading corpus {}...".format(corpus_file))
        corpus = {}
        with open(corpus_file) as inf:
            for line in inf:
                corpus.update(json.loads(line.strip()))
        term_vectors = {}
        for title, body in corpus.iteritems():
            term_vectors[title] = self.process_text(body)
        self.index = self.index_term_vectors(term_vectors)
        self.index_statistics = IndexStatistics(self.index, corpus)
        sys.stderr.write(" corpus loaded.\n")

    def load_stop_words(self, stop_words_file):
        """@param stop_words_file lists the stop words, one per line."""
        with open(stop_words_file) as inf:
            self.stop_words = set(w.strip() for w in inf.readlines())

    def print_index_statistics(self):
        print self.index_statistics

    def process_text(self, text):
        return list(chain(*map(self.filter, self.tokenize(text))))

    def tokenize(self, text):
        """Tokenizes @p text. Returns a list of tokens."""
        str_arr = list(text)
        for i in xrange(len(str_arr)):
            if not str_arr[i].isalnum():
                if not (str_arr[i] == '-' and
                        (str_arr[i - 1].isalnum() and str_arr[i + 1].isalnum())):
                    str_arr[i] = ' '
        return ''.join(str_arr).split()

    def filter(self, token):
        """
        Transforms tokens. Takes a token and returns a (possibly empty) list of
        terms.
        """
        # TODO
	token = token.strip(',.').lower()
	token = stem(token)
	if not token in self.stop_words:
		output = [token]
	else:
		output = []
	return output

    def index_term_vectors(self, term_vectors):
        """
        Creates an index from the per-document term vectors.

        Your input is a dictionary, which maps documents to the words it has,
        e.g.
        {
          "doc1": ["Sherlock", "see", "Sherlock"],
          "doc2": ["Watson", "Sherlock"]
        }

        Then, it should "invert" this dictionary to create the (inverted) index.
        The index is a term -> posting list dictionary, where the posting list
        is a list of Posting objects. Note that the object has a frequency field
        as well, so you have to count how many times the term is in the list.
        E.g. in our case:
        {
          "Sherlock": [Posting("doc1", 2), Posting("doc2", 1)],
          "Watson": [Posting("doc2", 1)],
          "see": [Posting("doc1", 1)]
        }

        @param term_vectors is a dictionary {doc_id: [terms]}.
                            The doc_id in this case is the original title.
        @return the index: a {term: [Posting]}.
        """
        # TODO
	output_dict = {}
	for key in term_vectors:
		for val in term_vectors[key]:
			if val in output_dict.keys():
				found = 0
				for x in output_dict[val]:
					if x.doc_id == key:
						x.tf = x.tf + 1
						found = 1
				if found == 0:
					p = Posting(key, 1)
					output_dict[val].append(p)
			else:
				output_dict[val] = [Posting(key, 1)]
		
        return output_dict

    def search(self, query_str):
        """The generic search method that parses a lisp-like QL."""
        try:
            query = nestedExpr().parseString(query_str)[0]
            documents = self.__search(query)
            return QueryResponse(
                sorted(documents, key=itemgetter(1), reverse=True))
        except Exception:
            return "Error: invalid query"

    def __search(self, query):
        """Recursively processes the query tree."""
        connective = query[0]
        subqueries = query[1:]
        subresults = []
        for sq in subqueries:
            if type(sq) is str or type(sq) is unicode:
                # Same filtering as for the documents
                plists = [[self.score(p, t) for p in self.index.get(t, [])]
                          for t in self.filter(sq)]
                subresults.append(self.join_posting_lists(plists, 'or'))
            else:
                subresults.append(self.__search(sq))
        return self.join_posting_lists(subresults, connective)

    def raw_text_search(self, query_str):
        """
        Matches the index agains the query. The query goes through the same
        tokenization and filtering process as the documents.
        """
        return self.search("(or {})".format(query_str))

    def score(self, posting, term):
        """Scores @p posting and returns a (doc_id, score) tuple."""
        score = self.similarity(term, posting.tf)
        return (posting.doc_id, score)

    def similarity(self, term, tf):
        """
        Computes the similarity score for a query and document. This simple
        implementation just returns 1.
        """
        # TODO
	return (1 + math.log(tf)) * math.log(self.index_statistics.no_docs/len(self.index[term]))

    def join_posting_lists(self, posting_lists, connective='or'):
        """
        Joins the already scored posting lists with the OR or AND logical
        connective. Returns a list of (doc_id, score) tuples.
        """
        if connective == 'and':
            return self.and_posting_lists(posting_lists)
        else:
            return self.or_posting_lists(posting_lists)

    def or_posting_lists(self, posting_lists):
        """
        Joins the already scored posting lists with the OR connective.

        @param posting_lists a list of [(doc_id, score)] tuple lists.
        @return a list of (doc_id, score) tuples.
        """
        ret = []
        for doc_id, postings in groupby(sorted(chain(*posting_lists),
                                               key=itemgetter(0)),
                                        key=itemgetter(0)):
            ret.append((doc_id, sum(post[1] for post in postings)))
        return ret

    def and_posting_lists(self, posting_lists):
        """
        Joins the already scored posting lists with the AND logical connective.
        Returns a list of (doc_id, score) tuples. Do not forget to sum the
        scores from the lists!

        For the algorithm, see Handout 8, pages 22-24.

        @param posting_lists a list of [(doc_id, score)] tuple lists.
        @return a list of (doc_id, score) tuples.
        """
        # TODO
	ret = []
	post1 = sorted(posting_lists[0], key=itemgetter(0))
	post2 = sorted(posting_lists[0], key=itemgetter(0))
	i = 0
	j = 0
	while i < len(post1) and j < len(post2):
		if post1[i][0] == post2[j][0]:	
			ret.append((post1[i][0], post1[i][1] + post2[j][1]))
			i = i+1
			j = j+1
		elif post1[i][0]< post2[j][0]:
			i = i+1
		else:
			j = j+1
	print ret
			
        return ret

def answer_queries(se, query_language=False):
    """Iterative query console."""
    print "Type queries ([Enter] to quit)"
    while True:
        sys.stdout.write("> ")
        q = sys.stdin.readline().strip()
        if q == "":
            break
        print se.search(q) if query_language else se.raw_text_search(q)
    print "bye."

def test_index(se, test_file):
    """Tests the search engine with queries from @p test_file."""
    with open(test_file) as inf:
        queries = [q.strip() for q in inf.readlines()]
    responses = [se.query(query) for query in queries]

def main():
    p = ArgumentParser()
    p.add_argument('-c', '--corpus', help='the corpus file', required=True)
    p.add_argument('-s', '--stop-words', help='the stop words file',
                   default=None)
    p.add_argument('-q', '--query-language', help='AND queries',
                   action='store_true')
    p.add_argument('-t', '--test', default=None,
                   help='test the index with the specified query file')
    args = p.parse_args()

    se = SearchEngine()
    if args.stop_words is not None:
        se.load_stop_words(args.stop_words)
    se.load_file(args.corpus)
    se.print_index_statistics()
    if args.test is not None:
        test_index(se, args.test)
    else:
        answer_queries(se, args.query_language)

if __name__ == '__main__':
    main()

