First result we obtained by splitting the channels and extracting the FLAIR channel only. The following picture is the result.

![Flair](https://github.com/user-attachments/assets/45e22606-2eb5-4ec5-a974-f0e6c2bc1601)

then bilateral and gaussian filters were applied to it separately

bilateral filtered 

![bilateral_filtered_image](https://github.com/user-attachments/assets/7ba54a1a-1d26-42a8-a563-ec7ff60cda9e)

gaussian filtered

![gaussian_filtered_image](https://github.com/user-attachments/assets/c52ac61a-9434-4c37-9d60-7a43b33a66b9)

each filtered image have went under either of two workflows, either canny edge filtering and then histogram based segmentaion (1) or canny edge filtering, otsu thresholding, morphological operation (gradient), and then histogram based segmentation (2)

canny edge filter for bilateral

![canny_edge_image](https://github.com/user-attachments/assets/b5e9353b-0bd3-4d21-b3da-e0cf48f74a50)

cleaned image after applying histogram based segmentation (1)

![Cleaned_Segmented_canHis](https://github.com/user-attachments/assets/52673ab6-145f-4132-a128-14c27a647c06)

canny edge filter for gaussian

![canny_edge_image (1)](https://github.com/user-attachments/assets/4d91b79d-4b5e-4473-a870-157f77e3b0b0)

cleaned image after applying histogram based segmentation (1)

![Cleaned_Segmented_canHis](https://github.com/user-attachments/assets/566fe673-22bb-44b9-8846-70521b968a3b)

otsu for bilateral

![otsu](https://github.com/user-attachments/assets/6062160b-bedd-46a3-aece-6ca77b07921a)

gradient for bilateral

![gradient](https://github.com/user-attachments/assets/b10bcecd-f9c8-4798-a5db-27fb4ab6b971)

histogram seg. (2)

![Cleaned_Segmented_gradHis](https://github.com/user-attachments/assets/912caa05-9f2b-4e56-a6ea-c5c7b677306b)

otsu for gaussian

![otsu (1)](https://github.com/user-attachments/assets/eec4ecff-924b-4c0a-bca1-be1a8735119a)

gradient for gaussian

![gradient (1)](https://github.com/user-attachments/assets/8fe03462-d06f-4600-90b0-3ec217497988)

histogram seg. (2)

![Cleaned_Segmented_gradHis](https://github.com/user-attachments/assets/26ea5c5a-2ba9-41f0-8d5e-fe433ff6070b)

