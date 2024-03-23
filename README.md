# Run Locally

## Prerequisites

- Python 3+
- MySql Database

## Create Virtual Environment

```bash
  python -m venv venv
  # Activate
  .\venv\Scripts\activate.bat
```
## Install Requirements

```bash
pip install -r .\requirments.txt
```
## Configure Properties

Open the file `config.properties` and configure all the required values.

## Run the app

```bash
python .\app.py
```

# Troubleshoot

**Issue**: Getting error `Data too long for column 'password_hash'`

**Resolution**:
  - Connect to MySql `retinopathy` (or configured database and run following query)

    ```sql
    ALTER TABLE retinopathy.`user` MODIFY COLUMN password_hash VARCHAR(300);
    ```
--------------
**Issue**: Not able to login

**Resolution**:
  - To be updated