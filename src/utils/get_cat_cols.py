import pyarrow as pa
import pyarrow.parquet as pq


def get_cat_cols(path):
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow

    cats = []

    for field in schema:
        is_cat_meta = False
        if field.metadata:  # dict[bytes, bytes]
            for k, v in field.metadata.items():
                kb = k or b""
                if b"CATEGORICAL" in kb:
                    is_cat_meta = True
                    break

        is_dict = pa.types.is_dictionary(field.type)

        if is_cat_meta or is_dict:
            cats.append(field.name)
    print(f"CATS: {len(cats)} columns")

    return cats