# Similar Image Search Using PyTorch

This project utilizes the EfficientNetB5 model for similar image search. By extracting features from images and storing them in a MySQL database, we can perform fast and efficient queries.

## Contents

1. [EfficientNetB5 Model](#efficientnetb5-model)
2. [Training the Model](#training-the-model)
3. [Feature Extraction](#feature-extraction)
4. [Storing Feature Vectors in MySQL](#storing-feature-vectors-in-mysql)
5. [Querying 10,000 Vectors](#querying-10000-vectors)
6. [k-Nearest Neighbors Search with kdTree](#k-nearest-neighbors-search-with-kdtree)
7. [Creating Server and Client](#creating-server-and-client)

## EfficientNetB5 Model

EfficientNetB5 is an advanced neural network model that optimizes performance and accuracy in image classification. This model is used to extract features from images that need to be searched.

![image](https://github.com/user-attachments/assets/6e0a219b-1359-4437-8b55-4a8a98f5f4e6)

## Training the Model

We train the EfficientNetB5 model on a large dataset with 10,000 images to enhance its classification capability. The training process uses PyTorch and related libraries.

![image_3](https://github.com/user-attachments/assets/0468f02a-093b-4bf7-860d-b760ea34748b)

We use the dataset from Kaggle at the following URL: [Kaggle Dataset](https://www.kaggle.com/datasets/utkarshsaxenadn/landscape-recognition-image-dataset-12k-images. "Kaggle Dataset")

## Feature Extraction

To extract features from images, we remove the classifier layer of the EfficientNetB5 model. This allows us to obtain feature vectors for each image.

```python
feature_extractor = torch.nn.Sequential(*list(model.children())[:-2])
```

When removing the final classifier layer, we obtain a feature vector of length 256.

## Storing Feature Vectors in MySQL

After extracting features, we store these vectors in a MySQL database for easy querying later.
```python
def save_vector_to_db(vector, connection, cursor, path_name):
    vector_json = json.dumps(vector)
    query = """
        INSERT INTO v6 
            (vector, path_name) VALUES (%s, %s);
    """
    cursor.execute(query, (vector_json, path_name,))
    connection.commit()
```

## Querying 10,000 Vectors

We have built a process to query and search through 10,000 vectors in the database, enabling users to find images quickly.

```python
def query_all_vectors(cursor, table_name):
    create_temp_table_query = f"""
        CREATE TEMPORARY TABLE temp_ids AS
        SELECT id
        FROM {table_name};
    """
        
    cursor.execute(create_temp_table_query)
    
    select_vectors_query = f"""
        SELECT id, vector, path_name
        FROM {table_name}
        WHERE id IN (SELECT id FROM temp_ids);
    """

    cursor.execute(select_vectors_query)
    
    list_vectors = cursor.fetchall()

    drop_temp_table_query = "DROP TEMPORARY TABLE temp_ids;"
    
    cursor.execute(drop_temp_table_query)
    
    return list_vectors
```

## k-Nearest Neighbors Search with kdTree

To find similar vectors, we use the kdTree algorithm. This algorithm allows for efficient searching in high-dimensional spaces.

#### Search Algorithm
1. Start at the Root: Begin searching from the root node of the kdTree.
2. Node Comparison: Compare the query point with the node’s point based on the current dimension. Depending on the comparison, proceed to the left or right subtree.
3. Track Nearest Neighbors: Maintain a list of the k closest points found during the search.
4. Backtrack if Necessary: If the distance to the splitting hyperplane is less than the current k-th nearest distance, check the opposite subtree.
5. Return Results: After exploring the necessary nodes, return the k nearest neighbors.

[![kdTree](https://opendsa-server.cs.vt.edu/ODSA/Books/winthrop/csci271/fall-2020/001/html/_images/KDtree.png "kdTree")](https://opendsa-server.cs.vt.edu/ODSA/Books/winthrop/csci271/fall-2020/001/html/_images/KDtree.png "kdTree")

#### Time Complexity
- Average Case: O(log n) for balanced trees.
- Worst Case: O(n) for unbalanced trees or degenerate cases.

## Creating Server and Client

We have developed a server and client that allow users to upload images and search for similar images. The server processes requests and returns results.

![image_5](https://github.com/user-attachments/assets/d3e70726-b03e-4169-ab3e-78b02568ec1a)

![image_6](https://github.com/user-attachments/assets/b9b46e6f-6976-490a-96bf-a5f3ca431298)

## How to Run the Project

1. Install the necessary libraries.
2. Train the model by running the training file.
3. Extract and store the vectors in MySQL.
4. Start the server and client.
5. Upload an image and perform a search.

## Contact

If you have any questions, please contact me at: [trung9204@gmail.com].

## License

This project is licensed under the [MIT License](LICENSE).
