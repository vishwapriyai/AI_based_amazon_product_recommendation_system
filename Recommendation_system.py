{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "LhuRflILh1kx"
   },
   "source": [
    "# IMPORT LIBRARIES"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "id": "fjV1TfKfhrLq"
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from surprise import SVD, KNNBasic, Dataset, Reader\n",
    "from surprise.model_selection import train_test_split\n",
    "from surprise.accuracy import rmse"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Icb1JLezh4FC"
   },
   "source": [
    "# READ DATA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 424
    },
    "id": "mvPin5dChWqp",
    "outputId": "fa62c4f3-61eb-4a5e-d152-9229ed7d5377"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>AKM1MP6P0OYPR</th>\n",
       "      <th>0132793040</th>\n",
       "      <th>5.0</th>\n",
       "      <th>1365811200</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>A2CX7LUOHB2NDG</td>\n",
       "      <td>0321732944</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1341100800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>A2NWSAGRHCP8N5</td>\n",
       "      <td>0439886341</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1367193600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>A2WNBOD3WNDNKT</td>\n",
       "      <td>0439886341</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1374451200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>A1GI0U4ZRJA8WN</td>\n",
       "      <td>0439886341</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1334707200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>A1QGNMC6O1VW39</td>\n",
       "      <td>0511189877</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1397433600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7824476</th>\n",
       "      <td>A2YZI3C9MOHC0L</td>\n",
       "      <td>BT008UKTMW</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1396569600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7824477</th>\n",
       "      <td>A322MDK0M89RHN</td>\n",
       "      <td>BT008UKTMW</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1313366400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7824478</th>\n",
       "      <td>A1MH90R0ADMIK0</td>\n",
       "      <td>BT008UKTMW</td>\n",
       "      <td>4.0</td>\n",
       "      <td>1404172800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7824479</th>\n",
       "      <td>A10M2KEFPEQDHN</td>\n",
       "      <td>BT008UKTMW</td>\n",
       "      <td>4.0</td>\n",
       "      <td>1297555200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7824480</th>\n",
       "      <td>A2G81TMIOIDEQQ</td>\n",
       "      <td>BT008V9J9U</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1312675200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>7824481 rows × 4 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "          AKM1MP6P0OYPR  0132793040  5.0  1365811200\n",
       "0        A2CX7LUOHB2NDG  0321732944  5.0  1341100800\n",
       "1        A2NWSAGRHCP8N5  0439886341  1.0  1367193600\n",
       "2        A2WNBOD3WNDNKT  0439886341  3.0  1374451200\n",
       "3        A1GI0U4ZRJA8WN  0439886341  1.0  1334707200\n",
       "4        A1QGNMC6O1VW39  0511189877  5.0  1397433600\n",
       "...                 ...         ...  ...         ...\n",
       "7824476  A2YZI3C9MOHC0L  BT008UKTMW  5.0  1396569600\n",
       "7824477  A322MDK0M89RHN  BT008UKTMW  5.0  1313366400\n",
       "7824478  A1MH90R0ADMIK0  BT008UKTMW  4.0  1404172800\n",
       "7824479  A10M2KEFPEQDHN  BT008UKTMW  4.0  1297555200\n",
       "7824480  A2G81TMIOIDEQQ  BT008V9J9U  5.0  1312675200\n",
       "\n",
       "[7824481 rows x 4 columns]"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data=pd.read_csv('Amazonproducts.csv')\n",
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 424
    },
    "id": "fYuMnYCfiZ6j",
    "outputId": "090444f8-313f-4d1e-f522-71d97c636116"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>userId</th>\n",
       "      <th>productId</th>\n",
       "      <th>Rating</th>\n",
       "      <th>timestamp</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>AKM1MP6P0OYPR</td>\n",
       "      <td>0132793040</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1365811200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>A2CX7LUOHB2NDG</td>\n",
       "      <td>0321732944</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1341100800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>A2NWSAGRHCP8N5</td>\n",
       "      <td>0439886341</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1367193600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>A2WNBOD3WNDNKT</td>\n",
       "      <td>0439886341</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1374451200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>A1GI0U4ZRJA8WN</td>\n",
       "      <td>0439886341</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1334707200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>59995</th>\n",
       "      <td>A3DX6U1B9KDUW4</td>\n",
       "      <td>B00004WHSD</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1121385600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>59996</th>\n",
       "      <td>A1BCHEPYUNRLJ</td>\n",
       "      <td>B00004WHSD</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1009152000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>59997</th>\n",
       "      <td>A1RLVKQQWHOQAW</td>\n",
       "      <td>B00004WHSD</td>\n",
       "      <td>5.0</td>\n",
       "      <td>984355200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>59998</th>\n",
       "      <td>A3T9DOOJ5B1U7O</td>\n",
       "      <td>B00004WHV7</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1014249600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>59999</th>\n",
       "      <td>A35FPXYK6GE49D</td>\n",
       "      <td>B00004WHV7</td>\n",
       "      <td>5.0</td>\n",
       "      <td>987292800</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>60000 rows × 4 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "               userId   productId  Rating   timestamp\n",
       "0       AKM1MP6P0OYPR  0132793040     5.0  1365811200\n",
       "1      A2CX7LUOHB2NDG  0321732944     5.0  1341100800\n",
       "2      A2NWSAGRHCP8N5  0439886341     1.0  1367193600\n",
       "3      A2WNBOD3WNDNKT  0439886341     3.0  1374451200\n",
       "4      A1GI0U4ZRJA8WN  0439886341     1.0  1334707200\n",
       "...               ...         ...     ...         ...\n",
       "59995  A3DX6U1B9KDUW4  B00004WHSD     1.0  1121385600\n",
       "59996   A1BCHEPYUNRLJ  B00004WHSD     5.0  1009152000\n",
       "59997  A1RLVKQQWHOQAW  B00004WHSD     5.0   984355200\n",
       "59998  A3T9DOOJ5B1U7O  B00004WHV7     5.0  1014249600\n",
       "59999  A35FPXYK6GE49D  B00004WHV7     5.0   987292800\n",
       "\n",
       "[60000 rows x 4 columns]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data=pd.read_csv('Amazonproducts.csv',names=['userId', 'productId','Rating','timestamp'])\n",
    "data=data.iloc[:60000,:]\n",
    "data.to_csv('Amazonproducts.csv', index=False)\n",
    "data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "SWGOPV3Th72p"
   },
   "source": [
    "# EDA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 206
    },
    "id": "XNFfXSszh7Mw",
    "outputId": "e639fd98-d7cb-485e-8e5c-598c29693a22"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>userId</th>\n",
       "      <th>productId</th>\n",
       "      <th>Rating</th>\n",
       "      <th>timestamp</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>AKM1MP6P0OYPR</td>\n",
       "      <td>0132793040</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1365811200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>A2CX7LUOHB2NDG</td>\n",
       "      <td>0321732944</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1341100800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>A2NWSAGRHCP8N5</td>\n",
       "      <td>0439886341</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1367193600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>A2WNBOD3WNDNKT</td>\n",
       "      <td>0439886341</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1374451200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>A1GI0U4ZRJA8WN</td>\n",
       "      <td>0439886341</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1334707200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           userId   productId  Rating   timestamp\n",
       "0   AKM1MP6P0OYPR  0132793040     5.0  1365811200\n",
       "1  A2CX7LUOHB2NDG  0321732944     5.0  1341100800\n",
       "2  A2NWSAGRHCP8N5  0439886341     1.0  1367193600\n",
       "3  A2WNBOD3WNDNKT  0439886341     3.0  1374451200\n",
       "4  A1GI0U4ZRJA8WN  0439886341     1.0  1334707200"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 206
    },
    "id": "q_tIllWbhp5o",
    "outputId": "9b35c5ac-480c-4d05-a20b-cb17dd8ae504"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>userId</th>\n",
       "      <th>productId</th>\n",
       "      <th>Rating</th>\n",
       "      <th>timestamp</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>59995</th>\n",
       "      <td>A3DX6U1B9KDUW4</td>\n",
       "      <td>B00004WHSD</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1121385600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>59996</th>\n",
       "      <td>A1BCHEPYUNRLJ</td>\n",
       "      <td>B00004WHSD</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1009152000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>59997</th>\n",
       "      <td>A1RLVKQQWHOQAW</td>\n",
       "      <td>B00004WHSD</td>\n",
       "      <td>5.0</td>\n",
       "      <td>984355200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>59998</th>\n",
       "      <td>A3T9DOOJ5B1U7O</td>\n",
       "      <td>B00004WHV7</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1014249600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>59999</th>\n",
       "      <td>A35FPXYK6GE49D</td>\n",
       "      <td>B00004WHV7</td>\n",
       "      <td>5.0</td>\n",
       "      <td>987292800</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               userId   productId  Rating   timestamp\n",
       "59995  A3DX6U1B9KDUW4  B00004WHSD     1.0  1121385600\n",
       "59996   A1BCHEPYUNRLJ  B00004WHSD     5.0  1009152000\n",
       "59997  A1RLVKQQWHOQAW  B00004WHSD     5.0   984355200\n",
       "59998  A3T9DOOJ5B1U7O  B00004WHV7     5.0  1014249600\n",
       "59999  A35FPXYK6GE49D  B00004WHV7     5.0   987292800"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "9O_HVPf6iG8P",
    "outputId": "4c49ad3f-cf2a-4ef2-c5f7-13de942bfa1e"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000, 4)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "5U_LGoM7iK0U",
    "outputId": "ea16aa02-554d-4632-c8ef-7a62a338d055"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "userId        object\n",
       "productId     object\n",
       "Rating       float64\n",
       "timestamp      int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "Wuoj137cipYs",
    "outputId": "389307ec-dc75-41ee-ed77-add4c57bab25"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of Rating: 60000\n",
      "Columns: ['userId' 'productId' 'Rating' 'timestamp']\n",
      "Number of Unique Users: 55334\n",
      "Number of Unique Products: 4140\n"
     ]
    }
   ],
   "source": [
    "print(\"Number of Rating: {}\".format(data.shape[0]) )\n",
    "print(\"Columns: {}\".format( np.array2string(data.columns.values)) )\n",
    "print(\"Number of Unique Users: {}\".format(len(data.userId.unique()) ) )\n",
    "print(\"Number of Unique Products: {}\".format(len(data.productId.unique())  ) )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "ZlFqYNTgj-j-",
    "outputId": "e651dc46-ee28-43ad-ee5c-f7a4471d4065"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    60000.000000\n",
       "mean         4.052350\n",
       "std          1.338559\n",
       "min          1.000000\n",
       "25%          4.000000\n",
       "50%          5.000000\n",
       "75%          5.000000\n",
       "max          5.000000\n",
       "Name: Rating, dtype: float64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.describe()['Rating']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "vasDbNQAkEX4",
    "outputId": "ab5f2e59-7694-4edf-bb39-884ee25660e1"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Minimum rating is: 1\n",
      "Maximum rating is: 5\n"
     ]
    }
   ],
   "source": [
    "print('Minimum rating is: %d' %(data.Rating.min()))\n",
    "print('Maximum rating is: %d' %(data.Rating.max()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "jifoP1PDkL2J",
    "outputId": "18fca502-b9ec-4518-91c5-e56e78f2ae3b"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of missing values in data: \n",
      " userId       0\n",
      "productId    0\n",
      "Rating       0\n",
      "timestamp    0\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "print('Number of missing values in data: \\n',data.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 368
    },
    "id": "szQr9gA_kVQE",
    "outputId": "9593c090-5c8e-4b4f-deb3-c6bcd3448d29"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'RATINGS COUNT')"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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/HE99NBDmjlzpi5fvqxLly5p9erVevbZZ9WrV68C08iRI3X+/Hl9/vnnJd7L9VfJCtv+5MmTdebMGQ0bNkxXr14tMF6csJJv6dKl+vDDDxUYGKguXboUWZeenm6+qOJ6V65cUXx8vJycnG7qCtuNuLi46NVXX9WhQ4cKvXroiF9//VUJCQny8fExX8lfp04dtWvXTp988onOnTtnV5+UlKSdO3fq6aefLm77kqRKlSrptddeU2pqqt54440C4zExMYqOjlZwcLA6dOhww22Fh4erWrVqioqKKpHeAMBq1nyQBABwxxgzZoxeeOEFRUdHq3r16jp//ryee+65Qms7dOig2rVra8mSJerdu3eJ99K/f3+9+eabhb6Yo1+/fkpJSdHUqVO1e/du9enTRw0aNFB2drZSUlK0bNkyVa1a1eHXw3/66aeqUqWKrly5op9//lmbNm3SV199pbZt22rVqlU3XPf48eN66KGH9MQTT6hLly7y8fHRqVOntGzZMn399dcKDw9XrVq1buUnKNTAgQMVGRmpt99+W927d//D+vxjMwxDJ06c0IIFC3T27FnNnz9fNpvNrHv33XcVHBysdu3aaeDAgapbt64OHTqkDz74QHXq1NGECRNK7BjGjx+v/fv36+2331ZiYqJ69uypihUraseOHfrkk0/UvHlzLVq06A+34+npqVGjRnHbIoAyg0AGAOVcjx49dN999+mf//ynmjdvLnd3dz355JOF1jo5OSkkJERLlizRL7/8opo1a5ZoLy4uLpo4caIGDRpU6Pg//vEPBQcHa/bs2froo4905swZVaxYUU2aNNGrr76q4cOHmx9B/iMjRoyQ9NuzVLVq1VK7du300UcfqV+/fgVedPJ7TZs21cyZM7VhwwbNnTtXGRkZcnd3V6tWrfTvf/9bQ4YMubkD/wMVK1bUyJEjNXnyZG3dulWPP/74Devzj02SKleurDZt2ujvf/+7XnjhBbu6zp0767///a/+9re/6b333tP58+fl7e2tfv36afLkySX6gWtnZ2etXLlSixcv1ocffqi//vWvunLliu677z5NmjRJr776qipXruzQtsLDwzVz5kxlZmaWWH8AYBWbURL3eAAAAAAAbhrPkAEAAACARQhkAAAAAGARAhkAAAAAWIRABgAAAAAWIZABAAAAgEUIZAAAAABgEb5DVkLy8vJ0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y5YtjZiYmNvbOO46gwcPNurVq2e4uroatWvXNrp06WKGMcPgPEPp+X0g41xDSendu7dRp04dw9XV1bjnnnuM3r17G0eOHDHHOddKH98hAwAAAACL8AwZAAAAAFiEQAYAAAAAFiGQAQAAAIBFCGQAAAAAYBECGQAAAABYhEAGAAAAABYhkAEAAACARQhkAAAAAGARAhkAAAAAWIRABgAo0wYOHCibzSabzaYKFSqoQYMGGjt2rC5fvmxXd/z4cbm6uqpVq1bmssmTJ5vrFjXl76N79+4F9vnWW2/Z7WPt2rXmOvkMw9C///1vBQYGysPDQ1WqVFHLli01atQoHTlyxKy7ePGiJkyYoPvuu0/u7u6qXbu2HnvsMX322Wcl9VMBACxAIAMAlHlPPfWUTp48qR9++EEzZszQ+++/r0mTJtnVREdH68UXX1RWVpZ27dolSXrttdd08uRJc/L19VVUVJTdsqK4u7vr7bff1tmzZ4usMQxD/fr101/+8hc988wz2rx5s7799lstWLBA7u7u+tvf/mbWDh8+XKtXr9a//vUvHT58WLGxserVq5d++eWXYv46AAAruVjdAAAApc3NzU0+Pj6SJD8/PwUFBSkuLk5vv/22pN+C0cKFCzV37lz5+vpqwYIFCggIUJUqVVSlShVzO87Ozqpataq5rRsJCgrSkSNHNHXqVE2bNq3QmhUrVmj58uX67LPP9Nxzz5nL7733XnXo0EGGYZjLPv/8c82aNUvPPPOMJKl+/fry9/e/+R8DAHBH4QoZAKBcSUlJUUJCglxdXc1lX375pS5evKigoCC99NJLWr58ubKzs4u1H2dnZ/3jH//Qv/71Lx0/frzQmmXLlqlp06Z2Yex619/e6OPjow0bNuj8+fPF6gsAcGchkAEAyrz169erSpUqcnd3V+vWrXXq1CmNGTPGHF+wYIH69OkjZ2dntWrVSg0bNtSqVauKvd8//elPateuXYHbI/N99913atq0qd2y8PBw88qcr6+vufyDDz5QQkKCatasqQcffFCjR4/WV199VeweAQDWIpABAMq8zp07Kzk5Wbt27VJoaKgGDRqknj17SpLOnTun1atX66WXXjLrX3rpJS1YsKBE9v32229r0aJFOnTokEP1b7zxhpKTkxUZGakLFy6Yyzt16qQffvhB8fHx6tWrlw4ePKiOHTvqzTffLJE+AQDWIJABAMq8ypUrq1GjRmrbtq0++ugj7dq1ywxcS5cu1eXLlxUQECAXFxe5uLho3Lhx2rFjh7777rti77tTp04KDg7WhAkTCow1btxYqampdstq166tRo0aycvLq0B9hQoV1LFjR40bN06bN29WVFSU3nzzTV25cqXYfQIArEEgAwCUK05OTnr99dc1ceJEXbp0SQsWLNCrr76q5ORkc/r666/VsWNHffTRRyWyz7feekvr1q1TYmKi3fK+ffsqNTX1ll9d36JFC127dq3AK/wBAHcPAhkAoNx54YUX5OzsrDlz5mjfvn0aOnSoWrVqZTf17dtXixYt0rVr14q9v9atW6t///5677337Jb36dNHvXr1Up8+fRQVFaVdu3bpxx9/1LZt27RixQo5OzubtY8//rjef/99JSUl6ccff9SGDRv0+uuvq3PnzvLw8Ch2jwAAaxDIAADljouLi0aOHKkJEyaofv36atasWYGaP/3pTzp16pQ2bNhQIvuMiopSXl6e3TKbzaYVK1Zo5syZ2rBhg7p06aKmTZtq8ODB8vPz044dO8za4OBgLVq0SF27dlXz5s31yiuvKDg4WCtXriyR/gAA1rAZ13/kBAAAAABw23CFDAAAAAAsQiADAAAAAIsQyAAAAADAIgQyAAAAALAIgQwAAAAALEIgAwAAAACLEMgAAAAAwCIEMgAAAACwCIEMAAAAACxCIAMAAAAAixDIAAAAAMAi/x/riF91MI184gAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 1000x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10,5))\n",
    "ratings = data['Rating'].value_counts()\n",
    "sns.barplot(x=ratings.index, y=ratings.values)\n",
    "plt.title('RATING DISTRIBUTION')\n",
    "plt.xlabel('RATINGS')\n",
    "plt.ylabel('RATINGS COUNT')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "lJCPQPPIlNyx",
    "outputId": "0b201853-82bf-4c3c-bcd6-fd481f82b0e9"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "userId\n",
       "A231WM2Z2JL0U3    45\n",
       "AY8Q1X7G96HV5     34\n",
       "A2BGZ52M908MJY    33\n",
       "A1NVD0TKNS1GT5    25\n",
       "A3MEIR72XKQY88    22\n",
       "A1MJMYLRTZ76ZX    22\n",
       "A1RPTVW5VEOSI     22\n",
       "ALUNVOQRXOZIA     21\n",
       "A243HY69GIAHFI    20\n",
       "A6FIAB28IS79      17\n",
       "A23ZO1BVFFLGHO    17\n",
       "A1ISUNUWG0K02V    16\n",
       "A7Y6AVS576M03     15\n",
       "A3IBOQ8R44YG9L    15\n",
       "A3FTI86WAVJOLG    14\n",
       "A1WVMDRJU19AFD    14\n",
       "ARXU3FESTWMJJ     13\n",
       "A2G2QNKDL1Y6AC    13\n",
       "A6ZPLVAUQ6695     13\n",
       "A3A15L96IYUO6V    13\n",
       "A2B7BUH8834Y6M    13\n",
       "A2M5GKAGV88LWD    13\n",
       "AD50TWQOM8W4G     12\n",
       "AEE2GJR0VF6R7     11\n",
       "A3PLX6PTM2ERKL    11\n",
       "Name: Rating, dtype: int64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "no_of_rated_products_per_user = data.groupby(by='userId')['Rating'].count().sort_values(ascending=False)\n",
    "no_of_rated_products_per_user.head(25)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 238
    },
    "id": "caVTwKbkoUt1",
    "outputId": "fdd49348-1f3d-4edc-a506-fd4f6976bf80"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Rating</th>\n",
       "      <th>ratings_count</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>userId</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>A001944026UMZ8T3K5QH1</th>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>A00766851QZZUBOVF4JFT</th>\n",
       "      <td>5.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>A01255851ZO1U93P8RKGE</th>\n",
       "      <td>5.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>A014623426J5CM7M12MBW</th>\n",
       "      <td>5.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>A01580702BRW77PSJ9X34</th>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                       Rating  ratings_count\n",
       "userId                                      \n",
       "A001944026UMZ8T3K5QH1     1.0              1\n",
       "A00766851QZZUBOVF4JFT     5.0              1\n",
       "A01255851ZO1U93P8RKGE     5.0              1\n",
       "A014623426J5CM7M12MBW     5.0              1\n",
       "A01580702BRW77PSJ9X34     1.0              1"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t2data_product_ratings =pd.DataFrame(data.groupby('userId')['Rating'].mean())\n",
    "t2data_product_ratings['ratings_count'] = pd.DataFrame(data.groupby('userId')['Rating'].count())\n",
    "t2data_product_ratings.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "id": "WrEgOly3p5Wh"
   },
   "outputs": [],
   "source": [
    "t2data_product_ratings['score'] = t2data_product_ratings['Rating']*t2data_product_ratings['ratings_count']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 477
    },
    "id": "SEbt3PLyql9g",
    "outputId": "1a98f8f5-4938-4f66-e381-71f9035e2adf"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.JointGrid at 0x13f0160ed10>"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 800x600 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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evTRRyVJQ4cO7RBkbrzxRi1fvlzPPvuszjjjDIXDYQWDQeXm5vb48Vpzc7Mef/xxHXfccZKkmTNn6u67726//5FHHtEdd9yhiy++WJL06KOP6s9//rPdywWAhBCSgDRy3nnnafHixTpw4IAefPBB9e7dW2VlZZKklpYW3XvvvXr22Wf16aefqqmpSY2NjcrNzY37PLm5ue0BSZIGDx6sPXv2SJIikYiqq6t1xhlntN/fq1cvjR8/Xq2trQmuEADsw8dtQBrp06ePRo4cqXHjxumpp57S2rVr9eSTT0qS7rvvPj300EO6/fbb9dprr2nDhg2aPHmympqa4j5PZmZmhz8HAgEZhmHLGgAgWQhJQJrKyMjQnXfeqR/+8Ic6ePCgVq9erYsuukhXXHGFxo0bp2OPPVYffPBBh+cEg0G1tLQkdN5wOKzCwkKtW7eu/baWlha98847CR0XAOxGSALS2Le+9S316tVLixYt0qhRo7RixQq9+eabev/993XDDTeourq6w+OPOeYYrV27Vp988olqamosfzx24403at68efrjH/+orVu36uabb9a+ffsUCATsWBYA2IKQBKSx3r17a+bMmfrZz36mW2+9VaeeeqomT56sc889V4MGDer0K/nf//731atXL40dO1ZHHXWUduzYYem8t99+uy677DJdddVVKikpUV5eniZPnqzs7GwbVgUA9ggYfFEAgMtaW1t1wgkn6NJLL9VPf/pTt8cBAEn8dhsAF/zjH//QX//6V02cOFGNjY169NFHtX37dn3nO99xezQAaMfHbQCSLiMjQ0uXLtXpp5+us88+W5s2bdIrr7yiE044we3RAKAdH7cBAACY4J0kAAAAE4QkAAAAE4QkAAAAE4QkAAAAE4QkAAAAE4QkAAAAE4QkAAAAE4QkAAAAE4QkAAAAE/8fH63VANSEd6QAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 600x600 with 3 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(8,6))\n",
    "sns.jointplot(x='Rating', y='ratings_count', data=t2data_product_ratings)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "kBrBWoUwqFWO"
   },
   "source": [
    "# DROP TIMESTAMP"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "id": "STaPJ3uRqp4I"
   },
   "outputs": [],
   "source": [
    "data.drop(['timestamp'], axis=1,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 424
    },
    "id": "xJ50606jqN1O",
    "outputId": "8c0123fc-281e-4e01-8981-e0b9c48bf2e7"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>userId</th>\n",
       "      <th>productId</th>\n",
       "      <th>Rating</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>AKM1MP6P0OYPR</td>\n",
       "      <td>0132793040</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>A2CX7LUOHB2NDG</td>\n",
       "      <td>0321732944</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>A2NWSAGRHCP8N5</td>\n",
       "      <td>0439886341</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>A2WNBOD3WNDNKT</td>\n",
       "      <td>0439886341</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>A1GI0U4ZRJA8WN</td>\n",
       "      <td>0439886341</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>59995</th>\n",
       "      <td>A3DX6U1B9KDUW4</td>\n",
       "      <td>B00004WHSD</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>59996</th>\n",
       "      <td>A1BCHEPYUNRLJ</td>\n",
       "      <td>B00004WHSD</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>59997</th>\n",
       "      <td>A1RLVKQQWHOQAW</td>\n",
       "      <td>B00004WHSD</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>59998</th>\n",
       "      <td>A3T9DOOJ5B1U7O</td>\n",
       "      <td>B00004WHV7</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>59999</th>\n",
       "      <td>A35FPXYK6GE49D</td>\n",
       "      <td>B00004WHV7</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>60000 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "               userId   productId  Rating\n",
       "0       AKM1MP6P0OYPR  0132793040     5.0\n",
       "1      A2CX7LUOHB2NDG  0321732944     5.0\n",
       "2      A2NWSAGRHCP8N5  0439886341     1.0\n",
       "3      A2WNBOD3WNDNKT  0439886341     3.0\n",
       "4      A1GI0U4ZRJA8WN  0439886341     1.0\n",
       "...               ...         ...     ...\n",
       "59995  A3DX6U1B9KDUW4  B00004WHSD     1.0\n",
       "59996   A1BCHEPYUNRLJ  B00004WHSD     5.0\n",
       "59997  A1RLVKQQWHOQAW  B00004WHSD     5.0\n",
       "59998  A3T9DOOJ5B1U7O  B00004WHV7     5.0\n",
       "59999  A35FPXYK6GE49D  B00004WHV7     5.0\n",
       "\n",
       "[60000 rows x 3 columns]"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "zG2DfMKdrJiM"
   },
   "source": [
    "# POPULARITY BASED METHOD"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "id": "PSfFzeSuqP_x"
   },
   "outputs": [],
   "source": [
    "new_df=data.groupby(\"productId\").filter(lambda x:x['Rating'].count() >=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 424
    },
    "id": "3GnmtPv0r4Cd",
    "outputId": "f5f299a7-e0bb-4c78-d114-2fa387fb222c"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>userId</th>\n",
       "      <th>productId</th>\n",
       "      <th>Rating</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>A2NWSAGRHCP8N5</td>\n",
       "      <td>0439886341</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>A2WNBOD3WNDNKT</td>\n",
       "      <td>0439886341</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>A1GI0U4ZRJA8WN</td>\n",
       "      <td>0439886341</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>A1QGNMC6O1VW39</td>\n",
       "      <td>0511189877</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>A3J3BRHTDRFJ2G</td>\n",
       "      <td>0511189877</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>59995</th>\n",
       "      <td>A3DX6U1B9KDUW4</td>\n",
       "      <td>B00004WHSD</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>59996</th>\n",
       "      <td>A1BCHEPYUNRLJ</td>\n",
       "      <td>B00004WHSD</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>59997</th>\n",
       "      <td>A1RLVKQQWHOQAW</td>\n",
       "      <td>B00004WHSD</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>59998</th>\n",
       "      <td>A3T9DOOJ5B1U7O</td>\n",
       "      <td>B00004WHV7</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>59999</th>\n",
       "      <td>A35FPXYK6GE49D</td>\n",
       "      <td>B00004WHV7</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>58715 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "               userId   productId  Rating\n",
       "2      A2NWSAGRHCP8N5  0439886341     1.0\n",
       "3      A2WNBOD3WNDNKT  0439886341     3.0\n",
       "4      A1GI0U4ZRJA8WN  0439886341     1.0\n",
       "5      A1QGNMC6O1VW39  0511189877     5.0\n",
       "6      A3J3BRHTDRFJ2G  0511189877     2.0\n",
       "...               ...         ...     ...\n",
       "59995  A3DX6U1B9KDUW4  B00004WHSD     1.0\n",
       "59996   A1BCHEPYUNRLJ  B00004WHSD     5.0\n",
       "59997  A1RLVKQQWHOQAW  B00004WHSD     5.0\n",
       "59998  A3T9DOOJ5B1U7O  B00004WHV7     5.0\n",
       "59999  A35FPXYK6GE49D  B00004WHV7     5.0\n",
       "\n",
       "[58715 rows x 3 columns]"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "xdOrZCeerhvm",
    "outputId": "096c8126-38dd-4251-8413-5f0e2584ebd3"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "productId\n",
       "0439886341    1.666667\n",
       "0511189877    4.500000\n",
       "0528881469    2.851852\n",
       "059400232X    5.000000\n",
       "0594012015    2.000000\n",
       "Name: Rating, dtype: float64"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new_df.groupby('productId')['Rating'].mean().head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "T79um76xr-18",
    "outputId": "be2ee109-16c8-4690-d436-d3b78716e8fb"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "productId\n",
       "B00004WHV7    5.0\n",
       "B000001OKZ    5.0\n",
       "B00003L65B    5.0\n",
       "9966308792    5.0\n",
       "B000000O3P    5.0\n",
       "Name: Rating, dtype: float64"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new_df.groupby('productId')['Rating'].mean().sort_values(ascending=False).head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "pprgE8g8sIp4",
    "outputId": "6e847afe-9ce2-46bb-8bcb-c640233c2cd7"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "productId\n",
       "B00001P4ZH    2075\n",
       "B00004T8R2    1692\n",
       "B00001WRSJ    1586\n",
       "0972683275    1051\n",
       "B00004SABB    1030\n",
       "Name: Rating, dtype: int64"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new_df.groupby('productId')['Rating'].count().sort_values(ascending=False).head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "id": "iAK2nKX0sOOC"
   },
   "outputs": [],
   "source": [
    "ratings_mean_count = pd.DataFrame(new_df.groupby('productId')['Rating'].mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "id": "7gWgM_TfsVcZ"
   },
   "outputs": [],
   "source": [
    "ratings_mean_count['rating_counts'] = pd.DataFrame(new_df.groupby('productId')['Rating'].count())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 238
    },
    "id": "OxLYBoWXsY3H",
    "outputId": "de6f05e4-97b3-45bb-bdd3-07bb52fc5da1"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Rating</th>\n",
       "      <th>rating_counts</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>productId</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0439886341</th>\n",
       "      <td>1.666667</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0511189877</th>\n",
       "      <td>4.500000</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0528881469</th>\n",
       "      <td>2.851852</td>\n",
       "      <td>27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>059400232X</th>\n",
       "      <td>5.000000</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0594012015</th>\n",
       "      <td>2.000000</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              Rating  rating_counts\n",
       "productId                          \n",
       "0439886341  1.666667              3\n",
       "0511189877  4.500000              6\n",
       "0528881469  2.851852             27\n",
       "059400232X  5.000000              3\n",
       "0594012015  2.000000              8"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ratings_mean_count.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 357
    },
    "id": "M_sparClsb0f",
    "outputId": "1bc31bf8-fb77-405f-fb50-b00e5d0bd882"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='productId'>"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "popular_products = pd.DataFrame(new_df.groupby('productId')['Rating'].count())\n",
    "most_popular = popular_products.sort_values('Rating', ascending=False)\n",
    "most_popular.head(30).plot(kind = \"bar\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 356
    },
    "id": "YICBS3rCso-4",
    "outputId": "a6da0ef4-79e1-48dc-c9fd-79b367846083"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='productId'>"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "popular_products = pd.DataFrame(new_df.groupby('productId')['Rating'].mean())\n",
    "most_popular = popular_products.sort_values('Rating', ascending=False)\n",
    "most_popular.head(30).plot(kind = \"bar\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "0TgOPhCWs5CP"
   },
   "source": [
    "# Collaberative filtering (Item-Item recommedation)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "dRR7ep2WtouT",
    "outputId": "ed01266f-3674-41ca-cbc9-c24d9d78344d"
   },
   "outputs": [],
   "source": [
    "#pip install scikit-surprise"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "id": "X6_0Y_6cs1TI"
   },
   "outputs": [],
   "source": [
    "from surprise import SVD,  SlopeOne\n",
    "from surprise import KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore\n",
    "\n",
    "from surprise import Dataset\n",
    "from surprise import accuracy\n",
    "from surprise import Reader\n",
    "import os\n",
    "from surprise.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "id": "b3UKjIJctYqE"
   },
   "outputs": [],
   "source": [
    "reader = Reader(rating_scale=(1, 5))\n",
    "data = Dataset.load_from_df(new_df,reader)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "id": "vRRZenySuYTd"
   },
   "outputs": [],
   "source": [
    "trainset, testset = train_test_split(data, test_size=0.3,random_state=10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ZwsVUYdHuvgW"
   },
   "source": [
    "# MEMORY BASED"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "id": "8R6H8ibdurG5"
   },
   "outputs": [],
   "source": [
    "bsl_options = {'method': 'als', 'n_epochs': 5, 'reg_u': 12, 'reg_i': 5 }\n",
    "sim_options={'name': 'pearson_baseline', 'user_based': False}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "k9YoMkx8ux93",
    "outputId": "0d4e8269-491c-4989-8043-2cb6adea8814"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Estimating biases using als...\n",
      "Computing the pearson_baseline similarity matrix...\n",
      "Done computing similarity matrix.\n",
      "RMSE: 1.3387\n"
     ]
    }
   ],
   "source": [
    "algo_KNNBasic = KNNBasic(k=5,sim_options = sim_options , bsl_options = bsl_options)\n",
    "predictions_KNNBasic = algo_KNNBasic.fit(trainset).test(testset)\n",
    "rmse_KNNBasic = accuracy.rmse(predictions_KNNBasic)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ayIi612q0_q4"
   },
   "source": [
    "# CREATE UNIQUE USER AND PRODUCTS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "vyQGsX2W00t_",
    "outputId": "1017e324-467f-4556-d848-9982bfe5d104"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['0439886341',\n",
       " '0511189877',\n",
       " '0528881469',\n",
       " '059400232X',\n",
       " '0594012015',\n",
       " '0594033896',\n",
       " '0594033926',\n",
       " '0594033934',\n",
       " '0594296420',\n",
       " '0594450209']"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "all_user_ids = list(new_df['userId'].unique())\n",
    "all_products = list(new_df['productId'].unique())\n",
    "all_products[:10]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "A-fkYaGW1g8H"
   },
   "source": [
    "# CHOOSE ONE USER"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "0EJ7C7GP1WhZ",
    "outputId": "5db2520c-04fc-456f-8706-b47dce99879c"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "54246"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(all_user_ids)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "25Ktb0C81Zgn",
    "outputId": "22b55f03-bc93-4455-e1e0-5336a6ad6fdb"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "User choosen to generate recommendation list is A3PRSCGIX3NY0X\n"
     ]
    }
   ],
   "source": [
    "user_index = 200\n",
    "uid = all_user_ids[user_index]\n",
    "print(\"User choosen to generate recommendation list is \" + str(uid))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "A8hsiRSJ1qda",
    "outputId": "e0a0d872-fe4d-4381-8fac-8ba74bcfeed0"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Choosen User has purchased the following items \n",
      "0972683275\n",
      "B00000JBUS\n",
      "Recommended items for user A3PRSCGIX3NY0X by KNNBasic \n",
      " ['B000028F42', 'B00001WRSJ', '1400599997', 'B00000JD4W', 'B00004WGNF', 'B00004SYO3', 'B00003006R', 'B00000JSGF', 'B00000J1V5', 'B00003WGP5', 'B00004S9AK', 'B00001P3XM', 'B00001P4ZH', 'B00004RC2D']\n"
     ]
    }
   ],
   "source": [
    "items_purchased = trainset.ur[trainset.to_inner_uid(uid)]\n",
    "\n",
    "\n",
    "print(\"Choosen User has purchased the following items \")\n",
    "for items in items_purchased[0]:\n",
    "    print(algo_KNNBasic.trainset.to_raw_iid(items))\n",
    "\n",
    "\n",
    "\n",
    "#getting K Neareset Neighbors for first item purchased by the choosen user\n",
    "KNN_Product = algo_KNNBasic.get_neighbors(items_purchased[0][0], 15)\n",
    "\n",
    "recommendedation_lits = []\n",
    "for product_iid in KNN_Product:\n",
    "    if not product_iid in items_purchased[0]: #user already has purchased the item\n",
    "        purchased_item = algo_KNNBasic.trainset.to_raw_iid(product_iid)\n",
    "        recommendedation_lits.append(purchased_item)\n",
    "print(\"Recommended items for user \" + str(uid) + \" by KNNBasic \\n\"  , recommendedation_lits)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {
    "id": "IBKHUolx-CgW"
   },
   "outputs": [],
   "source": [
    "df_predictions_KNNBasic = pd.DataFrame(predictions_KNNBasic, columns=['uid', 'iid', 'rui', 'est', 'details'])\n",
    "df_predictions_KNNBasic['err'] = abs(df_predictions_KNNBasic.est - df_predictions_KNNBasic.rui)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {
    "id": "6sfhsJLd-HLq"
   },
   "outputs": [],
   "source": [
    "best_predictions = df_predictions_KNNBasic.sort_values(by='err')[:10]\n",
    "worst_predictions = df_predictions_KNNBasic.sort_values(by='err')[-10:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 363
    },
    "id": "NxnflNbGEPjC",
    "outputId": "54c8d90e-757a-4cd5-e31d-6fbe943a3dce"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>uid</th>\n",
       "      <th>iid</th>\n",
       "      <th>rui</th>\n",
       "      <th>est</th>\n",
       "      <th>details</th>\n",
       "      <th>err</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>13473</th>\n",
       "      <td>A141E91QV31KER</td>\n",
       "      <td>B000001ON6</td>\n",
       "      <td>4.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>{'actual_k': 1, 'was_impossible': False}</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>620</th>\n",
       "      <td>A15U5NUS1EY7IQ</td>\n",
       "      <td>B000001OMI</td>\n",
       "      <td>5.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>{'actual_k': 1, 'was_impossible': False}</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16625</th>\n",
       "      <td>A1JJOV69MAU2J2</td>\n",
       "      <td>B00004TDN0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>{'actual_k': 1, 'was_impossible': False}</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11028</th>\n",
       "      <td>A2K5FK58JSWXJ9</td>\n",
       "      <td>B00004S9AK</td>\n",
       "      <td>5.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>{'actual_k': 1, 'was_impossible': False}</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12858</th>\n",
       "      <td>AYW1O00QM271D</td>\n",
       "      <td>B00004SB92</td>\n",
       "      <td>4.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>{'actual_k': 1, 'was_impossible': False}</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12815</th>\n",
       "      <td>A9U1BQILTMSMM</td>\n",
       "      <td>B00000J4FS</td>\n",
       "      <td>5.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>{'actual_k': 1, 'was_impossible': False}</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15211</th>\n",
       "      <td>A1ISUNUWG0K02V</td>\n",
       "      <td>B00004RC2D</td>\n",
       "      <td>4.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>{'actual_k': 1, 'was_impossible': False}</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17547</th>\n",
       "      <td>AQ28L0J970DVW</td>\n",
       "      <td>B00002EQCW</td>\n",
       "      <td>5.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>{'actual_k': 1, 'was_impossible': False}</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12303</th>\n",
       "      <td>A31LPS0TOFNNOT</td>\n",
       "      <td>B00004VX15</td>\n",
       "      <td>5.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>{'actual_k': 1, 'was_impossible': False}</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8103</th>\n",
       "      <td>A2OLQ18EOBZPE6</td>\n",
       "      <td>B00000J08C</td>\n",
       "      <td>5.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>{'actual_k': 2, 'was_impossible': False}</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                  uid         iid  rui  est  \\\n",
       "13473  A141E91QV31KER  B000001ON6  4.0  4.0   \n",
       "620    A15U5NUS1EY7IQ  B000001OMI  5.0  5.0   \n",
       "16625  A1JJOV69MAU2J2  B00004TDN0  4.0  4.0   \n",
       "11028  A2K5FK58JSWXJ9  B00004S9AK  5.0  5.0   \n",
       "12858   AYW1O00QM271D  B00004SB92  4.0  4.0   \n",
       "12815   A9U1BQILTMSMM  B00000J4FS  5.0  5.0   \n",
       "15211  A1ISUNUWG0K02V  B00004RC2D  4.0  4.0   \n",
       "17547   AQ28L0J970DVW  B00002EQCW  5.0  5.0   \n",
       "12303  A31LPS0TOFNNOT  B00004VX15  5.0  5.0   \n",
       "8103   A2OLQ18EOBZPE6  B00000J08C  5.0  5.0   \n",
       "\n",
       "                                        details  err  \n",
       "13473  {'actual_k': 1, 'was_impossible': False}  0.0  \n",
       "620    {'actual_k': 1, 'was_impossible': False}  0.0  \n",
       "16625  {'actual_k': 1, 'was_impossible': False}  0.0  \n",
       "11028  {'actual_k': 1, 'was_impossible': False}  0.0  \n",
       "12858  {'actual_k': 1, 'was_impossible': False}  0.0  \n",
       "12815  {'actual_k': 1, 'was_impossible': False}  0.0  \n",
       "15211  {'actual_k': 1, 'was_impossible': False}  0.0  \n",
       "17547  {'actual_k': 1, 'was_impossible': False}  0.0  \n",
       "12303  {'actual_k': 1, 'was_impossible': False}  0.0  \n",
       "8103   {'actual_k': 2, 'was_impossible': False}  0.0  "
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "best_predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 363
    },
    "id": "V3_BR83qERDy",
    "outputId": "d9d4961c-660a-4d55-b376-1561c6f37bf6"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>uid</th>\n",
       "      <th>iid</th>\n",
       "      <th>rui</th>\n",
       "      <th>est</th>\n",
       "      <th>details</th>\n",
       "      <th>err</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>12563</th>\n",
       "      <td>A34IOG3C3NLZ94</td>\n",
       "      <td>9573212919</td>\n",
       "      <td>1.0</td>\n",
       "      <td>4.058589</td>\n",
       "      <td>{'was_impossible': True, 'reason': 'User and/o...</td>\n",
       "      <td>3.058589</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4851</th>\n",
       "      <td>A2UKIGACNDA08D</td>\n",
       "      <td>B00004SB97</td>\n",
       "      <td>1.0</td>\n",
       "      <td>4.058589</td>\n",
       "      <td>{'was_impossible': True, 'reason': 'User and/o...</td>\n",
       "      <td>3.058589</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4852</th>\n",
       "      <td>A1S1MQMO5TC6RG</td>\n",
       "      <td>B00001P505</td>\n",
       "      <td>1.0</td>\n",
       "      <td>4.058589</td>\n",
       "      <td>{'was_impossible': True, 'reason': 'User and/o...</td>\n",
       "      <td>3.058589</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>A1K1YV2K1958SE</td>\n",
       "      <td>B00001OWYM</td>\n",
       "      <td>1.0</td>\n",
       "      <td>4.058589</td>\n",
       "      <td>{'was_impossible': True, 'reason': 'User and/o...</td>\n",
       "      <td>3.058589</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1423</th>\n",
       "      <td>A2E7WCIG181AMI</td>\n",
       "      <td>B00004TWVY</td>\n",
       "      <td>1.0</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>{'actual_k': 1, 'was_impossible': False}</td>\n",
       "      <td>4.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8183</th>\n",
       "      <td>A1YF0SKMGV2BIL</td>\n",
       "      <td>B00001P4ZH</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>{'actual_k': 1, 'was_impossible': False}</td>\n",
       "      <td>4.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9485</th>\n",
       "      <td>A2K1748WXNA8VJ</td>\n",
       "      <td>B00004VX39</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>{'actual_k': 1, 'was_impossible': False}</td>\n",
       "      <td>4.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12691</th>\n",
       "      <td>A3KN2OBPZ6QJLF</td>\n",
       "      <td>B00001P4ZH</td>\n",
       "      <td>1.0</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>{'actual_k': 1, 'was_impossible': False}</td>\n",
       "      <td>4.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3749</th>\n",
       "      <td>ABIU9UY9J2CAS</td>\n",
       "      <td>B000031KIM</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>{'actual_k': 1, 'was_impossible': False}</td>\n",
       "      <td>4.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15044</th>\n",
       "      <td>ALUNVOQRXOZIA</td>\n",
       "      <td>B00004SB92</td>\n",
       "      <td>1.0</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>{'actual_k': 1, 'was_impossible': False}</td>\n",
       "      <td>4.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                  uid         iid  rui       est  \\\n",
       "12563  A34IOG3C3NLZ94  9573212919  1.0  4.058589   \n",
       "4851   A2UKIGACNDA08D  B00004SB97  1.0  4.058589   \n",
       "4852   A1S1MQMO5TC6RG  B00001P505  1.0  4.058589   \n",
       "0      A1K1YV2K1958SE  B00001OWYM  1.0  4.058589   \n",
       "1423   A2E7WCIG181AMI  B00004TWVY  1.0  5.000000   \n",
       "8183   A1YF0SKMGV2BIL  B00001P4ZH  5.0  1.000000   \n",
       "9485   A2K1748WXNA8VJ  B00004VX39  5.0  1.000000   \n",
       "12691  A3KN2OBPZ6QJLF  B00001P4ZH  1.0  5.000000   \n",
       "3749    ABIU9UY9J2CAS  B000031KIM  5.0  1.000000   \n",
       "15044   ALUNVOQRXOZIA  B00004SB92  1.0  5.000000   \n",
       "\n",
       "                                                 details       err  \n",
       "12563  {'was_impossible': True, 'reason': 'User and/o...  3.058589  \n",
       "4851   {'was_impossible': True, 'reason': 'User and/o...  3.058589  \n",
       "4852   {'was_impossible': True, 'reason': 'User and/o...  3.058589  \n",
       "0      {'was_impossible': True, 'reason': 'User and/o...  3.058589  \n",
       "1423            {'actual_k': 1, 'was_impossible': False}  4.000000  \n",
       "8183            {'actual_k': 1, 'was_impossible': False}  4.000000  \n",
       "9485            {'actual_k': 1, 'was_impossible': False}  4.000000  \n",
       "12691           {'actual_k': 1, 'was_impossible': False}  4.000000  \n",
       "3749            {'actual_k': 1, 'was_impossible': False}  4.000000  \n",
       "15044           {'actual_k': 1, 'was_impossible': False}  4.000000  "
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "worst_predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "id": "ooHMbPJLESmD"
   },
   "outputs": [],
   "source": [
    "import pickle\n",
    "\n",
    "with open('knnbasic_model.pkl', 'wb') as f:\n",
    "    pickle.dump(algo_KNNBasic, f)"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "gpuClass": "standard",
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
