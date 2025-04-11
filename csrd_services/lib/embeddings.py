import numpy as np


def magnitude_vectors(vector):
    sum_of_squares = 0
    for x in vector:
        sum_of_squares += x * x
    return sum_of_squares**0.5


def cosine_of_vectors_embeddings(X, Y=None):
    if Y is None:
        Y = X

    cosine_similarity_matrix = []
    for x in X:
        row = []
        for y in Y:
            dot_product = sum(x[i] * y[i] for i in range(len(x)))
            row.append(dot_product)
        cosine_similarity_matrix.append(row)

    norm_X = [magnitude_vectors(x) for x in X]
    norm_Y = [magnitude_vectors(y) for y in Y]

    for i in range(len(X)):
        for j in range(len(Y)):
            cosine_similarity_matrix[i][j] /= norm_X[i] * norm_Y[j]

    return np.array(cosine_similarity_matrix)


def calculate_embeddings(model, combined_list, categories):
    results = []

    for query in combined_list:
        query_embedding_model_1 = model.encode([query])

        cos_sim_model_1 = {}
        for main_doc in categories:
            main_doc_embedding_model_1 = model.encode([main_doc])
            cos_sim_model_1[main_doc] = cosine_of_vectors_embeddings(
                query_embedding_model_1, main_doc_embedding_model_1
            ).flatten()[0]

        best_main_doc_model_1 = max(cos_sim_model_1, key=cos_sim_model_1.get)
        subdocs_model_1 = categories[best_main_doc_model_1]

        subdoc_embeddings_model_1 = model.encode(subdocs_model_1)
        cos_sim_model_1_sub = cosine_of_vectors_embeddings(
            query_embedding_model_1, subdoc_embeddings_model_1
        ).flatten()

        ranked_results_model_1 = np.argsort(cos_sim_model_1_sub)[::-1][:3]
        top_subdocs = [subdocs_model_1[idx] for idx in ranked_results_model_1]

        results.append(
            {
                "sentence": query,
                "category": best_main_doc_model_1,
                "subcategories": top_subdocs,
            }
        )

    return results
