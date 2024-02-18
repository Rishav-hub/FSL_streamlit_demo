import streamlit as st
import pandas as pd
from transformers import AutoProcessor, VisionEncoderDecoderModel
import requests
from PIL import Image
import torch
import warnings

from detect import model_inference
from utils import plot_bounding_boxes



warnings.filterwarnings('ignore')

uploaded_file = st.file_uploader("Upload your file here...", type=['png', 'jpeg', 'jpg'])
st.title("Firstsource Solutions Ltd. Document AI, Claim processing")


def run_prediction(image, model, processor):
    # prepare inputs
    pixel_values = processor(image, return_tensors="pt").pixel_values
    task_prompt = "<s>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=2,
        epsilon_cutoff=6e-4,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    prediction = processor.batch_decode(outputs.sequences)[0]
    prediction = prediction.replace("<one>", "1")
#     print("prediction:::", prediction)
    prediction = processor.token2json(prediction)
    return prediction, outputs


def split_and_expand(row):
    if row['Key'] == "33_Missing_Teeth":
        keys = [row['Key']]
        values = row['Value'].split(';')[0]
    else:
        keys = [row['Key']] * len(row['Value'].split(';'))
        values = row['Value'].split(';')
    return pd.DataFrame({'Key': keys, 'Value': values})

key_mapping = pd.read_excel("FSL_Forms_Keys.xlsx")
mapping_dict = key_mapping.set_index('Key_Name').to_dict()['Modified_key']
reverse_mapping_dict = {v: k for k, v in mapping_dict.items()}

@st.cache_resource
def load_model(device):
    try:
        processor = AutoProcessor.from_pretrained("Laskari-Naveen/ADA_98")
        model = VisionEncoderDecoderModel.from_pretrained("Laskari-Naveen/ADA_98").to(device)
        st.write("Model loaded successfully")
    except:
        print("Model Loading failed !!!")
    return processor, model


def convert_predictions_to_df(prediction):
    expanded_df = pd.DataFrame()
    result_df_each_image = pd.DataFrame()    
    each_image_output = pd.DataFrame(list(prediction.items()), columns=["Key", "Value"])
    expanded_df = pd.DataFrame(columns=['Key', 'Value'])

    for index, row in each_image_output[each_image_output['Value'].str.contains(';')].iterrows():
        expanded_df = pd.concat([expanded_df, pd.DataFrame(split_and_expand(row))], ignore_index=True)

    result_df_each_image = pd.concat([each_image_output, expanded_df], ignore_index=True)
    result_df_each_image = result_df_each_image.drop(result_df_each_image[result_df_each_image['Value'].str.contains(';')].index)

    for old_key, new_key in reverse_mapping_dict.items():
        result_df_each_image["Key"].replace(old_key, new_key, inplace=True)
    
    return result_df_each_image

@st.cache_data
def convert_df(df):
    return df.to_html().encode('utf-8')

if uploaded_file is not None:
    st.image(uploaded_file) 

    image = Image.open(uploaded_file).convert("RGB")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor, model = load_model(device)

    # ROI model
    st.write("Staring ROI Extraction")
    # print(uploaded_file.)
    fasterrcnn_result_df = model_inference(uploaded_file)
    plot_bounding_boxes(image, fasterrcnn_result_df, True)
    # st.dataframe(fasterrcnn_result_df, use_container_width=250)

    # Donut prediction
    prediction, output = run_prediction(image, model, processor)
    excel_df = convert_predictions_to_df(prediction)
    # html = convert_df(excel_df)
    # st.dataframe(excel_df, use_container_width=250)
    st.write("Donut Extraction Completed")

    # Add buttons to download dataframes
    st.download_button(
        label="Download ROI Output",
        data=fasterrcnn_result_df.to_csv(index=False).encode('utf-8'),
        file_name= f'{uploaded_file.name.split(".")[0]}_roi_result.csv',
        mime='text/csv'
    )

    st.download_button(
        label="Download Extraction Output",
        data=excel_df.to_csv(index=False).encode('utf-8'),
        file_name= f'{uploaded_file.name.split(".")[0]}_extraction_result.csv',
        mime='text/csv'
    )
    
