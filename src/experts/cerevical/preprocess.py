import pandas as pd

class PreprocessCervical:
    numeric_cols = [
        'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes',
        'Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives',
        'Hormonal Contraceptives (years)', 'IUD', 'IUD (years)', 'STDs', 'STDs (number)',
        'STDs:condylomatosis', 'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
        'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis', 'STDs:pelvic inflammatory disease',
        'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV',
        'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Time since first diagnosis',
        'STDs: Time since last diagnosis', 'Biopsy', 'Hinselmann', 'Schiller', 'Citology'
    ]
    
    std_cols = [
        'STDs:condylomatosis', 'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
        'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis', 'STDs:pelvic inflammatory disease',
        'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV',
        'STDs:Hepatitis B', 'STDs:HPV'
    ]
    
    test_cols = ["Hinselmann", "Schiller", "Citology", "Biopsy"]

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def transform(self) -> pd.DataFrame:
        numeric_existing = [c for c in self.numeric_cols if c in self.df.columns]
        self.df[numeric_existing] = self.df[numeric_existing].apply(pd.to_numeric, errors='coerce')

        std_existing = [c for c in self.std_cols if c in self.df.columns]
        self.df['total_std'] = self.df[std_existing].sum(axis=1) if std_existing else 0

        test_existing = [c for c in self.test_cols if c in self.df.columns]
        self.df['total_tests'] = self.df[test_existing].sum(axis=1) if test_existing else 0
        
        if 'Dx:Cancer' in self.df.columns:
            self.df = self.df.drop('Dx:Cancer', axis=1)

        self.df = self.df.fillna(0)

        return self.df
