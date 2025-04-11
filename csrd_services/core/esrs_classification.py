import os

from sentence_transformers import SentenceTransformer

from csrd_services.lib.embeddings import calculate_embeddings
from csrd_services.config import settings

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

ESRS_CATEGORIES = {
    "ESRS E1: Klimawandel – Berichterstattung über klimabezogene Risiken, Minderungsmaßnahmen, Übergangspläne zur Reduzierung von Treibhausgasemissionen (Fokus auf Emissionsreduktionen, Anpassungsstrategien und klimabezogene Risiken, z. B. Dekarbonisierungsziele, Einsatz erneuerbarer Energien, TCFD-konforme Klimaszenarioanalysen).": [
        "E1-1: Bewertung von Klimarisiken und Anfälligkeiten",
        "E1-2: Treibhausgas-Entnahmen und Kompensationen",
        "E1-3: Treibhausgas-Intensität pro Umsatz und Produkt",
        "E1-4: Aufschlüsselung des Verbrauchs erneuerbarer und nicht erneuerbarer Energie",
        "E1-5: Klima-Transformationsplan im Einklang mit dem Pariser Abkommen",
        "E1-6: Brutto-Treibhausgasemissionen (Scope 1, 2 und 3)",
        "E1-7: Finanzielle Auswirkungen klimabezogener Risiken und Chancen",
        "E1-8: Preisgestaltung für Kohlenstoff, einschließlich des internen Kohlenstoffpreissystems, Preis pro Tonne THG-Emissionen, Annahmen, Abdeckung der Bereiche 1, 2, 3 und Übereinstimmung mit Finanzberichten.",
        "E1-9: Vermögenswerte, die materiellen Klimarisiken (akuten und chronischen) ausgesetzt sind, Anpassungsmaßnahmen, finanziellen Auswirkungen, gestrandeten Vermögenswerten, Übergangsrisiken und monetarisierten THG-Emissionen mit Einsparpotenzial.",
    ],
    "ESRS E2: Umweltverschmutzung – Offenlegung von Maßnahmen, Richtlinien und Zielen zur Minimierung von Schadstoffemissionen und Verschmutzungen, Vermeidung oder Reduktion von Umweltauswirkungen sowie Verringerung von NOx- und SOx-Emissionen, Einsatz von Filteranlagen und Einhaltung von EU-Umweltnormen.": [
        "E2-1: Emissionen von Luftschadstoffen (SOx, NOx, Feinstaub, VOCs)",
        "E2-2: Einleitungen von Wasserverunreinigungen und Eindämmungsmaßnahmen",
        "E2-3: Abfallaufkommen und Management gefährlicher Abfälle",
        "E2-4: Maßnahmen zur Reduzierung chemischer und toxischer Substanzen",
        "E2-5: Einhaltung von Umweltvorschriften und Standards",
        "E2-6: Offenlegung der finanziellen Auswirkungen von Umwelt­risiken und -chancen, einschließlich Nettoumsatz aus Produkten mit bedenklichen Stoffen, OpEx, CapEx und Rückstellungen für Umweltschutz und Sanierungskosten.",
    ],
    "ESRS E3: Wasser- und Meeresressourcen – Berichterstattung über verantwortungsvollen Umgang mit Wasserressourcen und marinen Ökosystemen, inklusive nachhaltigem Wassermanagement, Wasserverbrauch, -qualität, Schutz mariner Lebensräume und deren nachhaltige Nutzung (z. B. Messung des Wasserverbrauchs, Investitionen in Wasseraufbereitungsanlagen, Schutz von Korallenriffen).": [
        "E3-1: Wasserentnahme und -verbrauch nach Quelle",
        "E3-2: Bewertung und Maßnahmen zur Minderung von Wasserstress",
        "E3-3: Auswirkungen auf Meeresökosysteme und Schutzmaßnahmen",
        "E3-4: Maßnahmen zur Kontrolle und Behandlung von Wasserverschmutzung",
        "E3-5: Strategien zur Wiederverwendung und Effizienzsteigerung von Wasser",
    ],
    "ESRS E4: Biodiversität und Ökosysteme – Darstellung von Maßnahmen zum Erhalt, zur Wiederherstellung und nachhaltigen Nutzung biologischer Vielfalt, natürlicher Lebensräume und Ökosystemleistungen (z. B. Aufforstungsprojekte, Schutz seltener Tier- und Pflanzenarten, Vermeidung von Eingriffen in sensible Ökosysteme).": [
        "E4-1: Flächennutzung und Überwachung der Abholzung",
        "E4-2: Auswirkungen auf Schutzgebiete und wichtige Biodiversitätszonen",
        "E4-3: Bewertung und Maßnahmen zur Minderung von Biodiversitätsrisiken",
        "E4-4: Wiederherstellungs- und Naturschutzprojekte",
        "E4-5: Berichterstattung über Biodiversitätsausgleichsmaßnahmen",
        "E4-6: Quantitative und qualitative Schätzungen der finanziellen Auswirkungen von Risiken in Bezug auf biologische Vielfalt und Ökosysteme, einschließlich Abhängigkeiten, Annahmen und Bewertungen von risikobehafteten Produkten und Dienstleistungen.",
    ],
    "ESRS E5: Ressourcennutzung und Kreislaufwirtschaft – Offenlegung von Strategien zur effizienten Ressourcennutzung, Abfallminimierung, Wiederverwendung, Recycling und Implementierung von Prinzipien der Kreislaufwirtschaft (z. B. Cradle-to-Cradle-Design, Einsatz recycelter Materialien, Rücknahmesysteme für Altgeräte).": [
        "E5-1: Materialeffizienz und nachhaltige Beschaffungsrichtlinien",
        "E5-2: Reduzierung des Abfallaufkommens und Recyclingquoten",
        "E5-3: Geschäftsmodelle und Initiativen zur Kreislaufwirtschaft",
        "E5-4: Bewertung und Minimierung der Ressourcenabhängigkeit",
        "E5-5: Lebenszyklusanalyse zentraler Produkte und Dienstleistungen",
        "E5-6: Offenlegung quantitativer und qualitativer Informationen zu den finanziellen Auswirkungen von Risiken und Chancen durch Ressourcennutzung und Kreislaufwirtschaft, einschließlich Annahmen, Zeithorizonten und Ressourceneffekten.",
    ],
    # Soziales (S) Datenpunkte
    "ESRS S1: Eigene Belegschaft – Informationen zu sozialen Aspekten für direkt beim Unternehmen beschäftigte Mitarbeitende, wie Arbeitsbedingungen, Vergütung, Arbeitsschutz, Aus- und Weiterbildung, Diversität und Inklusion (z. B. faire Lohnstrukturen, flexible Arbeitszeitmodelle, betriebliche Gesundheitsprogramme, Weiterbildungsinitiativen und Förderprogramme für Vielfalt).": [
        "S1-1: Demografie der Belegschaft (Alter, Geschlecht, Diversitätskennzahlen)",
        "S1-2: Arbeitsplatzsicherheit und Beschäftigungsverträge (befristet vs. unbefristet)",
        "S1-3: Unfallrate und Arbeitsschutzmaßnahmen",
        "S1-4: Schulung, Weiterqualifizierung und berufliche Entwicklung",
        "S1-5: Vergütung, Zusatzleistungen und Lohngleichheit",
        "S1-6: Merkmale der Belegschaft, Beschäftigtenzahl nach Geschlecht, Vertragsart, Regionen, Fluktuation, Methoden, Annahmen zu Kopfzahl, Vollzeitäquivalenten und Beschäftigungskontext, mit Aufschlüsselung nach Geschlecht und Region.",
        "S1-7: Zahl der Nichtbeschäftigten, einschließlich Selbstständiger und von Arbeitsvermittlungen gestellter Personen. Aufschlüsselung nach Arbeitsart, Unternehmensbeziehung und Schätzmethoden, mit relevanten Hintergrundinformationen.",
        "S1-8: Prozentsatz der Beschäftigten unter Kollektivvereinbarungen, einschließlich spezifischer Regionen und Länder. Vereinbarungen für nicht gewerkschaftlich organisierte Beschäftigte und sozialer Dialogumfang.",
        "S1-9: Geschlechterverteilung auf der obersten Führungsebene, prozentualer Anteil, Altersverteilung der Beschäftigten mit Kopfzahl, sowie Unternehmensdefinition von Topmanagement.",
        "S1-10: Informationen, ob alle Beschäftigten eine angemessene Entlohnung gemäß Referenzwerten erhalten, einschließlich Länder, in denen Löhne unter diesen Werten liegen.",
        "S1-11: Absicherung der Beschäftigten durch öffentliche Programme oder Leistungen bei Krankheit, Arbeitslosigkeit, Unfällen, Behinderungen, Elternurlaub und Ruhestand. Aufschlüsselung nach Beschäftigtenarten und Ländern.",
        "S1-12: Prozentsatz der Beschäftigten mit Behinderungen, nach Geschlecht aufgeschlüsselt, vorbehaltlich rechtlicher Einschränkungen bei der Datenerhebung. Kontextinformationen zur Interpretation der Daten.",
        "S1-13: Indikatoren für Ausbildung und Qualifikationen, nach Geschlecht: Teilnahme an Leistungsbeurteilungen und Schulungen, durchschnittliche Stundenanzahl, nach Kategorien und Geschlecht.",
        "S1-14: Prozentsatz der Beschäftigten, die durch das Gesundheits- und Sicherheitsmanagementsystem des Unternehmens abgedeckt sind. Daten zu Arbeitsunfällen, Todesfällen und krankheitsbedingten Ausfällen, einschließlich Audits.",
        "S1-15: Prozentsatz der Beschäftigten mit Anspruch auf familiären Urlaub und der Anspruchsberechtigten, die diesen Urlaub in Anspruch genommen haben, nach Geschlecht. Bestätigung des Anspruchs auf familiären Urlaub.",
        "S1-16: Geschlechtsspezifische Lohnlücke und Verhältnis der Vergütung zwischen dem höchstbezahlten und dem Median der Beschäftigten. Aufschlüsselung nach Beschäftigtenkategorie, Land und Vergütungskomponenten, angepasst an Kaufkraftunterschiede.",
        "S1-17: Anzahl gemeldeter Diskriminierungsfälle, einschließlich Belästigung und Beschwerden über interne sowie nationale Kanäle. Geldstrafen, Bußgelder und Entschädigungen, sowie Menschenrechtsverletzungen und ergriffene Abhilfemaßnahmen."
    ],
    "ESRS S2: Beschäftigte in der Wertschöpfungskette – Berichterstattung über Sozialstandards für Beschäftigte bei Lieferanten und Partnern, z. B. faire Arbeitsbedingungen, Vermeidung von Kinder- oder Zwangsarbeit, Einhaltung sozialer Mindeststandards entlang der gesamten Lieferkette, Lieferantenaudits zu Arbeits- und Sozialstandards und Berücksichtigung von ILO-Standards.": [
        "S2-1: Sorgfaltspflicht in Bezug auf Menschenrechte und Risikominderung",
        "S2-2: Faire Löhne und Arbeitsbedingungen in der Lieferkette",
        "S2-3: Gesundheits- und Sicherheitsvorfälle in Zulieferbetrieben",
        "S2-4: Risikoabschätzungen für Kinder- und Zwangsarbeit",
        "S2-5: ESG-Compliance-Überwachung von Lieferanten",
    ],
    "ESRS S3: Betroffene Gemeinschaften – Offenlegung der Auswirkungen unternehmerischer Aktivitäten auf lokale Gemeinschaften, Anwohner, indigene Gruppen und NGOs, einschließlich Dialog, Stakeholder-Engagement, Menschenrechtsaspekten und sozialer Infrastruktur (z. B. Stakeholder-Dialoge mit Anwohnern vor Bauprojekten, Investitionen in lokale Bildungs- oder Gesundheitsinitiativen, Wahrung der Rechte indigener Völker).": [
        "S3-1: Gemeinschaftsbeteiligung und Wirkungsmessung",
        "S3-2: Soziale Investitionen und Beiträge zur lokalen Entwicklung",
        "S3-3: Menschenrechtsauswirkungen auf betroffene Gemeinschaften",
        "S3-4: Zugang zu grundlegenden Dienstleistungen und Infrastruktur",
        "S3-5: Konsultationsprozesse mit Stakeholdern",
    ],
    "ESRS S4: Verbraucherinnen und Endverbraucher – Darstellung von Themen wie Produktsicherheit, verantwortungsvolles Marketing, Schutz von Kundendaten, Kundenzufriedenheit und fairen Vertragsbedingungen für Personen, die Produkte oder Dienstleistungen des Unternehmens konsumieren (z. B. klare Produktkennzeichnung, Beschwerdemechanismen für Endverbraucher, Datenschutzmaßnahmen).": [
        "S4-1: Produktsicherheitsvorfälle und Einhaltung gesetzlicher Vorschriften",
        "S4-2: Schutz der Verbraucherdaten und Datenschutzmaßnahmen",
        "S4-3: Ethik im Marketing und faire Werbung",
        "S4-4: Mechanismen zur Bearbeitung von Kundenbeschwerden",
        "S4-5: Barrierefreiheit und Inklusion von Produkten und Dienstleistungen",
    ],
    # Governance (G) Datenpunkte
    "ESRS G1: Geschäftspraktiken – Informationen zu Unternehmensführung, Ethik, Compliance, Anti-Korruptionsmaßnahmen, Transparenz und Corporate-Governance-Strukturen (z. B. Verhaltenskodizes für Führungskräfte, Whistleblower-Systeme, transparente Vorstandsentscheidungen, interne Kontrollmechanismen).": [
        "G1-1: Struktur, Zusammensetzung und Diversität des Vorstands",
        "G1-2: Vergütung von Führungskräften, Anreize und Transparenz",
        "G1-3: Ethikrichtlinien und Maßnahmen zur Korruptionsbekämpfung",
        "G1-4: Schutz von Hinweisgebern, Meldekanäle und Compliance",
        "G1-5: Risikomanagementrahmen, ESG-Aufsicht und interne Kontrollen",
        "G1-6: ESG-Governance-Verantwortlichkeiten auf Vorstands- und Führungsebene",
    ],
}

def get_model():
    """Initialize the sentence transformer model with proper error handling"""
    try:
        model = SentenceTransformer(
            'paraphrase-multilingual-mpnet-base-v2',
            cache_folder=CACHE_DIR,
            token=settings.HUGGINGFACE_TOKEN
        )
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model: {str(e)}. Please check your internet connection and HuggingFace token.")


def classify_text(texts: list[str]):
    """
    Classify text content into the most relevant ESRS categories.
    """
    model = get_model()

    results = calculate_embeddings(model, texts, ESRS_CATEGORIES)

    return results
