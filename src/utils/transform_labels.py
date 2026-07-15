#!/usr/bin/env python
"""
Transform entity labels to a standardized set of categories.

Dictionary for mapping generated entity labels to the four main categories:
Tobacco, Cannabis, Alcohol, Drugs. This module provides a function to normalize
entity labels based on a predefined mapping.
"""

# --- Local Imports ---
from .logging import logger
from ..imports import (
    shutil,
    Path,
    ABBREVIATIONS_MAPPING,
    AUGMENTATION_PATH,
    CORRECTED_PATH
)

# --- Constants ---

ENTITIES_MAPPING = {
    "Tobacco": [
        # Different variations of tobacco-related terms, including common misspellings and artifacts from model generation.
        "Tobacco", "tobacco", "Tabaco", "Tabacco", "Toabcco", "Toabacco", "_Tobacco",
        "vaping", "vape", "vapeo", "Vaping", "Vape", "vape:Tobacco", "vaping:Tobacco",
        "vapor", "v[Tobacco", "pipa:Tobacco", "pipa", "Pipe", "Pipes", "Pipe/Tobacco", 
        "Fumar", "Fumo", "Cigarette", "Cigarettes/year", "Cigarrillos", "Nicotina", 
        "Nicotine", "Paquete", "Puros", "Tabaquismo", "VAPING",
        "Fumador", "Fumadora", "Fumadores", "Fumadoras", "Fumadores/as",
        "E-cigarette", "E-cigarettes", "E-cigarrillos:vaping", "\"vaping\"", "Vaporización",
        "Vaporizador", "Vaporizadores", "Vaporizer", "Shisha",
        "Cigaretos", "Cigarettes", "Cigarrillo", "Cigarilla", "Cigarillo", "Cigarrete", 
        "Cigarretes", "Cigarrets", "Cigarro", "Cigarros:forma", "Cigarrotype",
        "Fuma", "Smokes", "Smoker", "Smoking", "Smoking-amount", "Smoking-duration", 
        "Smoking-method", "Smoking-onset", "Smoking_status", "Tabaco,fumador", "Tobacco,",
        "Ex-fumadora", "Ex-fumer", "Fumer", "Tabáquico", "Tobaco", "Exfumador", "Non-smoker",
        "@:tobacco", "[fumador:tobacco", "[tabaco", "[tabaco:drug", "[tabaquismo:tobacco", 
        "[exfumador:tobacco", "[exfumadora:tobacco", "[fumador:", "[fumadora:tobacco", "Ex-[fumador:tobacco"
    ],
    
    "Cannabis": [
        # Different variations of cannabis-related terms, including common misspellings and artifacts from model generation.
        "Cannabis", "cannabis", "Cannabinoids", "Marihuana", "Marihuana:Cannabis", 
        "Porro", "Porros", "THC", "Cannabidiol", "Cannabinoides", "Cannabis\"",
        "Cannabis,", "Cannabis:", "Cannabinoid", "Cannabis”",
        "Cb", "Cbd", "Cbda", "Cbnd", "Hashish", "Hachís", "Háchís",
        "Marihuana,", "Marijuana", "Tetrahydrocannabinol", "Thc:cannabis", "Cannabis-few",
        "[cannabis:cannabis", "Tabaco-cannabis" # Clasificado aquí al ser una mezcla cuyo foco principal suele ser el psicoactivo
    ],
    
    "Alcohol": [
        # Different variations of alcohol-related terms, including common misspellings and artifacts from model generation.
        "Alcohol", "alcohol", "Acohol", "Alcohol,", "Alcoholism", "Cerveza", "Cerveza,",
        "Etanol", "Vino", "Alcohole", "Alcoholic", "Alcohols", "Alcohólicos", "Drink",
        "Drinks", "Drinking", "Alcool", "Beer", "Beers", "Spirits", "Wine",
        "Alcohol-occasionally", "Álcool", "Alcoholicbeverage", "Ethanol", "Methanol",
        ":alcohol", "[alcohol:alcohol", "Comercial:alcohol", "Alcohols:alcohol", "Cerveza:alcohol", 
        "[oh:alcohol", "A[alcoholemia:alcohol", "[alcohol", "[alcohol:", "[bebedor:alcohol"
    ],
    
    "Drugs": [
        # Different variations of drug-related terms, including common misspellings and artifacts from model generation.
        "Drug", "Drogas", "drogas", "Drog", "a:Drogas", "Drug}", "leftover:Drug", "Jey:Drug",
        "Caffeine", "caffeine", "Caffeine:Drug", "Psychotropic", "PsychotropicMedication", 
        "Psychoactive", "PsychoactiveDrug", "Psychoactive_Substance", "PsychoactiveCompound", 
        "Amphetamines", "Rohypnol", "Lorazepam", "Bropuprenorphine", "Poppy", "Narcótico", 
        "Narcotic", "Narcotics", "Psychedelics", "substance", "sustancia", "toxicológico",
        "Droga", "droga", "Drogas:Sindependencia", "OTRAS_DROGAS", "OtraDroga", "Otras-Drogas", 
        "Otras_Drogas", "Otras_drogas", "Otros_Drogas", "Sustancia", "Sustancias", "sustancias",
        "Cafeína", "Cafés", "Energizante", "Estimulante", "Estimulantes", "Estimulantes,", 
        "Stimulante", "Stimulantes", "Stimulants", "Vitesse", "Nergetic", "Metilfenidato",
        "Anfetaminas", "Cocaine", "Cocaína", "Crack", "Cristal", "Mefaedrona", "3-MMC", "Roca",
        "Heroin", "Herosina", "Heroína", "Héroina", "Fentanyl", "Codeine", "Morphine", "Metadona", 
        "Methadone", "Buprenorfina/naloxona", "Buprenorphine/naloxone", "Tramadol", "Tapentadol", 
        "Opiate", "Opiates", "Opioid", "Opioides", "Opioids", "Opiáceo", "Opiáceos", "Opiáceos,", 
        "Opiáceos:Sindependencia", "Agonista/Antagonista", "Agonista/antagonista", 
        "Agonistas/Antagonistas", "Agonistas/antagonistas", "Naloxona:Droga",
        "Ketamina", "Ketamine", "LSD", "Peyote", "Psilocybe", "Psilocybin", "GHB",
        "Psicofármaco", "Psicofármaco/Stimulante", "Psicofármacos", "Psicotrópico", "Psychotropic_drugs",
        "psicóticos", "antipsicóticos", "Sedante", "Alprazolam", "BZD", "Baclofeno", "Benzo", 
        "Benzodiacepinas", "Benzodiazepinas", "Benzodiazepines", "benzodiacepinas", "Clonazepam", 
        "Clorazepate", "Chlorpromazine", "Citalopram", "Duloxetina", "Escitalopram", "Haloperidol", 
        "Lithium", "Litio", "Olanzapina", "Risperidona", "Risperidone", "Topiramate", "Valproate", 
        "Valproato", "Valproico", "Venlafaxine", "esnifar",
        "Drug\"", "Drug)", "Drug:forma-informal", "Drogas:drug", "Tóxica:psychotropic", 
        "Psychotropics", "Toxic", "Toxicos", "Toxicity", "Toxin", "Substancias", "Illicitsubstance", 
        "Polysubstance", "Sustancia-multi", "Sustancias-multi", "Sustancias-multi:drug", "[drug:drug", 
        "Toxhabits", "Sustancia:", "Otras-sustancias", "Otras_dropas", "Sniffed:othrdrogs", 
        "Weekly:othrdrogs", "Toxicological", "Toxicómana", "Addictive", "Nps",
        "Cafeinas", "Caffeina", "Caffeína", "Caffeína:drug", "Café", "Caféine", "Caféína", "Coffee", "Tea", 
        "[café:drug", "Energy", "Amphetamine", "Amfetamina", "Amfetaminas", "Amfetamine", "Anfetamina", 
        "Anfetamines", "Metanfetamina", "Methamphetamine", "Metametfetamina", "Metamfetamina", 
        "Methylfenida", "Methilfenidato", "3-methylmethcathinone", "4-metylcathinone", "Mefedrone", 
        "Mdpv", "Pentedrona", "Crystal", "Khat", "Pseudoefedrina", "Betel", "Stimulantdrug",
        "Cocaina", "Cocinea", "Cocaine:drug", "[cocaine:drug", "[cocaína:droga", "Levamisole",
        "Psilocibina", "Psychedelic", "[psilocibina:drug", "[dpta:psilocybin", "[lsd:", "Lysergic", 
        "Alucinógeno", "Alucinógenos", "[alucinógenos:drug", "Hallucinogen", "Hallucinogens", "Ayahuasca", 
        "Mescalina", "Mdba", "Mdma", "Methylenedioximetanfetamina", "Methylenedioxy-n-metilamfetamina", 
        "Methylenedioxymethamphetamine", "Éxtasis", "Gamma-hidroxibutirato", "Ghb:drug", "Gbl", "Poppers",
        "Buprenex", "Buprenorfina", "Buprenorfina/opioid", "Buprenorfine", "Buprenorfon", "Buprenorphina", 
        "Buprenorphine", "Buprenorphine,", "Buprenorphine:opioid", "Fentanilo", "Heroina", "Heroine", 
        "Héroína", "Methadona", "Methadone,", "Morfina", "Oxicodona", "Opiatos", "Opiaceos", "Agon-opi", 
        "Agonis-antagonistas", "Agonist-antagonist:drug", "Agonista/antagonistas", "Agonista/opiáceo", 
        "Agonistas-antagonistas", "Agonistas/antagonistas_opiáceos", "Agonistas/antagonistasopiáceos", 
        "Agonistasantagonistasopiaceos", "Agonistasantagonistasopiáceos", "Agonists-opiáceos", 
        "Agonists/antagonistas", "Agonists/antagonists", "Antagonista/agonista_opiáceos", 
        "Antagonista/opioide", "Endogenousopioid", "Opioidagnistsaantagonists", "Naloxona:buprenorfine", 
        "Naloxone", "Naltrexona", "Naltrexone", "[heroína:drug", "[buprenorfina/naloxona:", 
        "[buprenorfina:drug", "Kratom", "Mitragyna",
        "Amitriptilina", "Ansiolíticos", "Antidepresivo", "Antidepressants", "Antidepressivo", 
        "Antiepiléptico", "Antiepileptics", "Antipsychotics", "Antipsicotic", "Antipsicótico", "Anxiolítico", 
        "Benzodiazepina", "Benzodiazepina:drug", "Benzodiazepine", "Bzd:drug", "Clozapina", "Diazepam", 
        "Fluoxetine", "Isrs", "Mirtazapina", "Pregabalin", "Sertralina", "Venlafaxina", "Zaleplon,", 
        "Sedative", "Sedatives", "Hypnotics", "Gabapentina", "Barbitúricos:drug", "Analgesic", "Analgésicos",
        "Psico-fármacos", "Psicofarma", "Psych-drugs", "Psych_farm", "Psych_farmac", "Psychf", "Psychfarm", 
        "Psychfarma", "Psychfarmacocos", "Psychfármaco", "Psychfármacos", "Psycho", "Psychofarm", 
        "Psychofarmaco", "Psychofarmacos", "Psychofármaco", "Psychofármacos", "Psychopharm", "Psychopharmac", 
        "Psychopharmac:antipsicótico", "Psychopharmaceutical", "Psychopharmaceuticals", "Psychopharmaco", 
        "Psychopharmacologic", "Psychopharmacologicalagent", "Psychopharmacology", "Psychopharmacon", 
        "Psychopharmacos", "Psychopharmacs", "Psychopioid", "Psychoscópicos", "Psychotropic_medications", 
        "Psychotropicdrug", "Anabolic-androgenic", "Anabólico", "Anabólicos", "Anabolizante", "Anabolizantes"
    ]
}

# Invert the dictionary to create a direct mapping for normalization.
DIRECT_MAPPING = {}
for correct_entity, generated_list in ENTITIES_MAPPING.items():
    for generated_entity in generated_list:
        # Map the generated entity to the correct entity in lowercase for consistency.
        DIRECT_MAPPING[generated_entity.lower()] = correct_entity.lower()

# --- Helper Functions ---

def entity_normalization(
    original_entity: str
) -> str:
    """
    Normalizes a generated entity to one of the four main categories: Tobacco, Cannabis, Alcohol, Drugs.

    Args:
        original_entity (str): The entity generated by the model.
    
    Returns:
        str: The normalized entity. If the original entity is not in the mapping, it returns the original entity in lowercase.
    """
    # Use the direct mapping to normalize the entity, defaulting to the original entity if not found.
    mapped = DIRECT_MAPPING.get(original_entity.lower(), original_entity.lower())
    return str(mapped).capitalize()


# --- Main Function ---

def transform_labels_function(
    methods: list[str],
    models: list[str],
    input_base_dir: Path = Path(f"./{AUGMENTATION_PATH}/"),
) -> None:
    """
    Transforms the entity labels in .ann files for each method and model combination.

    Args:
        methods (List[str]): List of augmentation methods.
        models (List[str]): List of model names.
    """
    # Iterate through each method and model combination to process the corresponding .ann files.
    for method in methods:
        method_abbr = ABBREVIATIONS_MAPPING.get(method, method)
        for model in models:
            model_abbr = model.lower().replace(".", "")
            logger.info(f"\n\n--- Analyzing entities for method '{method_abbr}' and model '{model_abbr}' ---")
            
            # Define input and output directories based on method and model.
            input_dir = Path(f"{input_base_dir}/{method_abbr}/{model_abbr}/")
            output_dir = Path(f"./{CORRECTED_PATH}/{method_abbr}/{model_abbr}/")
            incorrect_entities = set()

            # Process each .ann file in the input directory.
            for ann_file in input_dir.glob("*.ann"):
                with open(ann_file, "r", encoding="utf-8") as f:
                    corrected_entities = []
                    for line in f:
                        if line.startswith("T"):
                            parts = line.strip().split("\t")
                            if len(parts) > 1:
                                entity_type = parts[1].split()[0]
                                start_index = parts[1].split()[-2]
                                end_index = parts[1].split()[-1]
                                corrected_type = entity_normalization(entity_type)
                                corrected_entities.append(f"{parts[0]}\t{corrected_type} {start_index} {end_index}\t{'\t'.join(parts[2:])}")

                                # Track any entities that are not one of the four main categories.
                                if corrected_type not in ["Tobacco", "Cannabis", "Alcohol", "Drugs"]:
                                    incorrect_entities.add(corrected_type)

                    # Save the corrected entity types to a new .ann file in the output directory.
                    output_file = output_dir / ann_file.name
                    output_dir.mkdir(parents=True, exist_ok=True)
                    with open(output_file, "a", encoding="utf-8") as out_f:
                        for entity in corrected_entities:
                            out_f.write(f"{entity}\n")
                    
                    # Copy the corresponding .txt file to the output directory to maintain the document pair.
                    txt_file = ann_file.with_suffix(".txt")
                    if txt_file.exists():
                        shutil.copy(txt_file, output_dir / txt_file.name)

            # Print any incorrect entities found during processing.
            logger.info("\nIncorrect entities found (not mapped to Tobacco, Cannabis, Alcohol, Drugs):")
            for entity in sorted(incorrect_entities):
                logger.info(entity)
