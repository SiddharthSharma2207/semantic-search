from pydantic_settings import BaseSettings,SettingsConfigDict


class Settings(BaseSettings):
    api_key:str
    llm_model_name:str
    collection_name:str
    base_url:str
    chroma_db_dir:str
    model_config = SettingsConfigDict(env_file="/Users/siddharthsharma/PycharmProjects/climba/.env")