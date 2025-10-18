{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d4d21992",
   "metadata": {},
   "outputs": [],
   "source": [
    "import subprocess\n",
    "import sys\n",
    "subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"pandas\", \"openpyxl\"])\n",
    "import pandas as pd\n",
    "file_path = r'C:\\Users\\mcobp\\OneDrive\\Desktop\\DataBreach\\Data_Breach_Chronology.xlsx'\n",
    "data = pd.read_excel(file_path)\n",
    "print(data.head())\n",
    "print(\"\\nColumn names:\")\n",
    "print(data.columns.tolist())\n",
    "print(f\"\\nShape: {data.shape[0]} rows, {data.shape[1]} columns\")\n",
    "print(\"\\nBasic statistics:\")\n",
    "print(data.describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a554841e",
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
