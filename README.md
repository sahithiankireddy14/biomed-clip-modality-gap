# Modality Gap

Here, we study the latent structure of text and image embeddings for Biomed-CLIP. 

Initially, I examined the pairwise Euclidean distance between image and text embeddings. We can see that the according image-text pairs tend to have the smallest euclidean distance, as intutively expected. However, it's important to note that the latent vectors of Biomed-CLIP reside on a 512-dimensional hypersphere, so Euclidean distance is a less intuitive measure. Consequently, I analyzed the cosine similarity distribution between all text pairs and image pairs.  I expected the matching image-text pairs (1st and 4th quadrants) to have the highest cosine similarity, but as seen in the below heatmap, the observed results suggest otherwise. This highlights an persisting modality gap, where daa of the same domain is closest in space rather than data with the same context. Furthermore, the highest similarity is found within the text class, indicating that the representations for text labels lack distinctiveness despite their disparate medical contexts.
<img width="887" alt="cosine-similarity-heatmap" src="https://github.com/sahithiankireddy14/biomed-clip-modality-gap/assets/46660955/7b9dfb19-4069-4523-8f5a-23fb2c790e41">



The modality gap is further illustratred in the t-SNE plot below. Corresponding image-text pairs are color-coded. Strikingly, instead of corresponding image-text pairs appearing in close proximity, all images are close together, and then all labels are clustered together in space.

<img width="568" alt="tsne" src="https://github.com/sahithiankireddy14/biomed-clip-modality-gap/assets/46660955/7c57a1e1-26a7-46e9-9c9e-bb9720c77f82">


This prompts the natural research question of determining how to build stronge representations within the latent space. It's imperative to establish geometric consistency between data of the same features rather than the same domain. Some ideas I have include altering the CLIP loss function to penalize terms that are not part of the according image-text pairs. 

### Research Question:
Can we build geometric consistency between the same features and effectively reduce the modality gap?
