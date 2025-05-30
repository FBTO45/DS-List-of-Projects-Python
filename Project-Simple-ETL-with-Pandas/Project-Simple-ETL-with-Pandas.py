import pandas as pd

# Load dataset
df_participant = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/dqthon-participants.csv')

# Ekstraksi postal code dari address
df_participant['postal_code'] = df_participant['address'].str.extract(r'(\d+)$')

# Ekstraksi nama kota dari address
df_participant['city'] = df_participant['address'].str.extract(r'(?<=\n)(\w.+)(?=,)')

# Buat kolom GitHub profile dari first_name dan last_name
df_participant['github_profile'] = (
    'https://github.com/' +
    df_participant['first_name'].str.lower() +
    df_participant['last_name'].str.lower()
)

# Membersihkan phone number
df_participant['cleaned_phone_number'] = df_participant['phone_number'].str.replace(r'^(\+62|62)', '0', regex=True)
df_participant['cleaned_phone_number'] = df_participant['cleaned_phone_number'].str.replace(r'[()-]', '', regex=True)
df_participant['cleaned_phone_number'] = df_participant['cleaned_phone_number'].str.replace(r'\s+', '', regex=True)

# Buat kolom team_name
def create_team_name(col):
    abbrev_name = "%s%s" % (col['first_name'][0], col['last_name'][0])
    country = col['country']
    abbrev_institute = ''.join([word[0] for word in col['institute'].split()])
    return "%s-%s-%s" % (abbrev_name, country, abbrev_institute)

df_participant['team_name'] = df_participant.apply(create_team_name, axis=1)

# Buat kolom email
def create_email(col):
    first_name_lower = col['first_name'].lower()
    last_name_lower = col['last_name'].lower()
    institute = ''.join([word[0] for word in col['institute'].lower().split()])
    
    if 'Universitas' in col['institute']:
        if len(col['country'].split()) > 1:
            country = ''.join([word[0] for word in col['country'].lower().split()])
        else:
            country = col['country'][:3].lower()
        return "%s%s@%s.ac.%s" % (first_name_lower, last_name_lower, institute, country)
    
    return "%s%s@%s.com" % (first_name_lower, last_name_lower, institute)

df_participant['email'] = df_participant.apply(create_email, axis=1)

# Ubah kolom birth_date menjadi datetime
df_participant['birth_date'] = pd.to_datetime(df_participant['birth_date'], format='%d %b %Y')

# Ubah kolom register_time menjadi datetime
df_participant['register_at'] = pd.to_datetime(df_participant['register_time'], unit='s')

df_participant.to_csv("cleaned_participants.csv", index=False)
# atau
print(df_participant.head())
