import email
import imaplib
import os

import asyncio

from genaitor.core import (
    Orchestrator, Flow, ExecutionMode
)
from genaitor.presets.agents import structure_data_agent
import pandas as pd
import json
from genaitor.core.base import OCRImageAgent
from sqlalchemy import create_engine
import re

columns_map = {
    'volts': ['volts', 'volt'],
    'amps': ['amps', 'amp'],
    'rpm': ['rpm', 'fl_rpm'],
    'efficiency': ['nema_nom_eff', 'guaranteed_min_eff', 'nom_eff', 'efficiency', 'nema_nom_eff_'],
    'phase': ['phase', 'ph'],
    'frame': ['frame', 'frame_number'],
    'service_factor': ['sf', 'service_factor', 'ser_f'],
    'serial_number': ['serial_number', 'sn'],
    'hz': ['hz', 'hertz'],
    'rating': ['rating', 'time_rating'],
    'code': ['code', 'kva_code'],
    'insulation_class': ['class_insul', 'class', 'ins'],
    'ambient_temp': ['amb', 'amb_temp', 'max_amb'],
    'enclosure': ['enclosure', 'encl'],
    'bearing_de': ['sh_end_brg', 'drive_end_bearing', 'brg_de', 'bearings_de'],
    'bearing_ode': ['opp_end_brg', 'opp_de_bearing', 'no_ode', 'ode'],
    'model_number': ['model_number', 'cat_number'],
}

def unify_columns(df, map):
    for new, old in map.items():
        col_found = [col for col in old if col in df.columns]
        if col_found:
            df[new] = df[col_found].bfill(axis=1).iloc[:, 0]
            df.drop(columns=[col for col in col_found if col != new], inplace=True)
    return df

def clean_name(col):
    col = col.lower()
    col = re.sub(r'[^\w\s]', '', col)
    col = re.sub(r'\s+', '_', col)
    col = col.replace('no', 'number')
    col = col.replace('f_l', 'fl')
    col = col.replace('serialnumber', 'serial_number')
    col = col.replace('modelnumber', 'model_number')
    col = col.replace('catnumber', 'model_number')
    col = col.replace('3phase', 'phase')
    return col.strip()

def get_emails():
    EMAIL = 'yanbarrosyan@gmail.com'
    PASSWORD = 'qfkj lvjf dmjt gasf'
    IMAP_SERVER = 'imap.gmail.com'

    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL, PASSWORD)
    mail.select('inbox')

    _, data = mail.search(None, 'ALL')
    mail_ids = data[0].split()
    latest_email_id = mail_ids[-1]

    _, data = mail.fetch(latest_email_id, '(RFC822)')
    imgs_path = []
    for response_part in data:
        if isinstance(response_part, tuple):
            msg = email.message_from_bytes(response_part[1])
            mail_from = msg['from']
            mail_subject = msg['subject']

            print(f'📥 From: {mail_from}')
            print(f'📩 Subject: {mail_subject}')

            email_body = ""

            os.makedirs('attachments', exist_ok=True)
            os.makedirs('emails', exist_ok=True)

            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))

                if content_type == "text/plain" and "attachment" not in content_disposition:
                    charset = part.get_content_charset() or 'utf-8'
                    body = part.get_payload(decode=True).decode(charset, errors='replace')
                    email_body += body

                if part.get_content_maintype() == 'image' or 'attachment' in content_disposition:
                    filename = part.get_filename()
                    if filename:
                        file_data = part.get_payload(decode=True)
                        with open(f'attachments/{filename}', 'wb') as f:
                            f.write(file_data)
                        imgs_path.append(f'attachments/{filename}')
                
    return email_body, imgs_path

def combine_columns(df, map):
    for new_column, sinonms in map.items():
        columns = [col for col in df.columns if col in sinonms]
        if columns:
            df[new_column] = df[columns].bfill(axis=1).iloc[:, 0]
            df.drop(columns=[col for col in columns if col != new_column], inplace=True)
    return df

def clean_column_name(col):
    col = col.lower()
    col = re.sub(r'[^\w\s]', '', col)
    col = re.sub(r'\s+', '_', col)
    col = col.replace('no', 'number')
    col = col.replace('f_l', 'fl')
    return col.strip()

def unify_motor_columns(df):
    df.columns = [clean_column_name(col) for col in df.columns]

    column_map = {
        'volts': ['volts'],
        'amps': ['amps', 'amp'],
        'rpm': ['rpm'],
        'hz': ['hz', 'hertz'],
        'phase': ['phase', '3phase', 'ph'],
        'efficiency': ['efficiency', 'nema_numberm_eff', 'numberm_eff'],
        'frame': ['frame', 'frame_number'],
        'service_factor': ['sf', 'service_factor', 'ser_f'],
        'serial_number': ['serial_number', 'sn'],
        'model_number': ['model_number', 'cat_number'],
        'code': ['code', 'kva_code'],
        'insulation_class': ['class_insul', 'class', 'ins', 'insulation_class'],
        'rating': ['rating', 'time_rating'],
        'enclosure': ['enclosure', 'encl', 'encl_'],
        'bearing_de': ['sh_end_brg', 'drive_end_bearing', 'brg_de', 'bearings_de', 'bearing_de'],
        'bearing_ode': ['opp_end_brg', 'opp_de_bearing', 'no_ode', 'ode', 'bearing_ode'],
        'ambient_temp': ['amb', 'amb_temp', 'ambtemp', 'max_amb', 'ambient_temp'],
        'nema_efficiency': ['nema_numberm_eff', 'nema_nom_eff', 'nema_nom_eff_'],
    }

    for unified_col, variants in column_map.items():
        present = [col for col in variants if col in df.columns]
        if len(present) > 1:
            df[unified_col] = df[present].bfill(axis=1).iloc[:, 0]
            df.drop(columns=[col for col in present if col != unified_col], inplace=True)
        elif len(present) == 1 and unified_col != present[0]:
            df.rename(columns={present[0]: unified_col}, inplace=True)

    df = df.loc[:, ~df.columns.duplicated()]

    return df

async def main():
    print("\nInitializing OCR Agents...")
    email_body, imgs_path = get_emails()
    print(f"[Email Body]\n{email_body}\n[Files attached]\n{len(imgs_path)}")
    ocr_agent = OCRImageAgent(api_key="AIzaSyCpVoMqVI9pCfCDDj1_1DgZdXsglG5olfo")
    imgs_txt = []
    for img_path in imgs_path:
        img_txt = await ocr_agent.process_request(img_path)        
        imgs_txt.append(img_txt)        
    
    orchestrator = Orchestrator(
        agents={
            "structure_data_agent": structure_data_agent, 
            },
        flows={
            "structure_data_flow": Flow(agents=["structure_data_agent"], context_pass=[True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )
    input_data = '\n-------\n'.join(str(item) for item in imgs_txt)

    try:
        result = await orchestrator.process_request(input_data, flow_name='structure_data_flow')
        
        if result["success"]:
            structured_data = result['content']['structure_data_agent'].content.strip()
            cleaned_data = structured_data.replace('```json\n','').replace('```','')
            engine = create_engine("mysql+mysqlconnector://root:1234@localhost:3306/ocr_agent")
            df = pd.DataFrame(json.loads(cleaned_data))
            df.columns = [clean_name(col) for col in df.columns]
            df = unify_columns(df, columns_map)
            df = unify_motor_columns(df)
            df.to_excel("motors_nameplates.xlsx", index=False)
            print("Data stored on table motors_nameplates")
            print("Check the ")
            
        else:
            print(f"\nError: {result['error']}")
    
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())