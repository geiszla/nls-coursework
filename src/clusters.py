

class Clusters():
    def __init__(self, target_words: List[str], cluster_count: int, context_size: int):
        # Initiate a 2D numpy array for the word x word matrix filled with zeroes
        word_x_word_matrix = numpy.zeros((len(target_words), len(self.vocabulary)), numpy.int32)

        target_word_to_index = {word: index for index, word in enumerate(target_words)}
        vocabulary_word_to_index = {word: index for index, word in enumerate(self.vocabulary)}

        print('Building co-occurrence array...')
        for line in self.lines:
            for index, word in enumerate(line):
                context = get_word_context(index, line, context_size)

                if word in target_words:
                    context_indexes = [vocabulary_word_to_index[context_word]
                        for context_word in context]

                    word_x_word_matrix[target_word_to_index[word], context_indexes] += 1

        self.clusters = KMeans(cluster_count).fit_predict(word_x_word_matrix)
