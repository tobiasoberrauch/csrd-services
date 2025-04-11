import unittest

from csrd_services.core.esrs_classification import classify_text


class TestESRSCategorization(unittest.TestCase):

    def test_each_category_assignment(self):
        texts_examples = {
            "E1-1: Brutto-Treibhausgasemissionen (Scope 1, 2 und 3)": "Unsere Brutto-Treibhausgasemissionen betrugen im letzten Jahr 150.000 Tonnen CO2.",
            "E1-2: Treibhausgas-Entnahmen und Kompensationen": "Wir haben Klimakompensationen in Höhe von 10.000 Tonnen CO2 realisiert.",
            "E1-3: Treibhausgas-Intensität pro Umsatz und Produkt": "Die Treibhausgasintensität betrug 20 kg CO2 pro verkauftem Produkt.",
            "E1-4: Aufschlüsselung des Verbrauchs erneuerbarer und nicht erneuerbarer Energie": "Unser Energieverbrauch umfasst 60% erneuerbare und 40% nicht erneuerbare Energien.",
            "E1-5: Klima-Transformationsplan im Einklang mit dem Pariser Abkommen": "Unser Klima-Transformationsplan sieht eine Emissionsreduzierung von 50% bis 2030 vor.",
            "E1-6: Bewertung von Klimarisiken und Anfälligkeiten": "Die Bewertung ergab, dass 25% unserer Anlagen klimarisikoanfällig sind.",
            "E1-7: Finanzielle Auswirkungen klimabezogener Risiken und Chancen": "Klimabezogene Risiken könnten finanzielle Auswirkungen von bis zu 5 Millionen Euro haben.",
            "E2-1: Emissionen von Luftschadstoffen (SOx, NOx, Feinstaub, VOCs)": "Emissionen von Luftschadstoffen beliefen sich auf 200 Tonnen SOx und NOx.",
            "E2-2: Einleitungen von Wasserverunreinigungen und Eindämmungsmaßnahmen": "Wasserverschmutzung wurde um 90% reduziert.",
            "E2-3: Abfallaufkommen und Management gefährlicher Abfälle": "Im letzten Jahr entsorgten wir 500 Tonnen gefährliche Abfälle fachgerecht.",
            "E2-4: Maßnahmen zur Reduzierung chemischer und toxischer Substanzen": "Wir reduzierten chemische Substanzen in der Produktion um 30%.",
            "E2-5: Einhaltung von Umweltvorschriften und Standards": "Wir erfüllten 100% der geltenden Umweltvorschriften.",
            "E3-1: Wasserentnahme und -verbrauch nach Quelle": "Unser Wasserverbrauch lag bei 300.000 Kubikmetern aus kommunalen Quellen.",
            "E3-2: Bewertung und Maßnahmen zur Minderung von Wasserstress": "Wir konnten den Wasserstress in Produktionsstätten um 40% reduzieren.",
            "E3-3: Auswirkungen auf Meeresökosysteme und Schutzmaßnahmen": "Maßnahmen schützten erfolgreich 100 km² Meeresökosysteme.",
            "E3-4: Maßnahmen zur Kontrolle und Behandlung von Wasserverschmutzung": "Wasserverschmutzung wurde um 90% reduziert.",
            "E3-5: Strategien zur Wiederverwendung und Effizienzsteigerung von Wasser": "Wir erhöhten die Wasserwiederverwendung auf 75%.",
            "E4-1: Flächennutzung und Überwachung der Abholzung": "Wir überwachen 1000 Hektar Fläche, um Abholzung zu verhindern.",
            "E4-2: Auswirkungen auf Schutzgebiete und wichtige Biodiversitätszonen": "Wir schützen aktiv 50 Biodiversitätszonen.",
            "E4-3: Bewertung und Maßnahmen zur Minderung von Biodiversitätsrisiken": "Risiken für die Biodiversität wurden um 60% reduziert.",
            "E4-4: Wiederherstellungs- und Naturschutzprojekte": "Wir führen aktuell 15 Naturschutzprojekte durch.",
            "E4-5: Berichterstattung über Biodiversitätsausgleichsmaßnahmen": "Es wurden 200 Biodiversitätsausgleichsmaßnahmen dokumentiert.",
            "E5-1: Materialeffizienz und nachhaltige Beschaffungsrichtlinien": "Unsere Materialeffizienz stieg um 25%.",
            "E5-2: Reduzierung des Abfallaufkommens und Recyclingquoten": "Recyclingquote verbesserte sich auf 80%.",
            "E5-3: Geschäftsmodelle und Initiativen zur Kreislaufwirtschaft": "Wir implementierten 10 Kreislaufwirtschaftsmodelle.",
            "E5-4: Bewertung und Minimierung der Ressourcenabhängigkeit": "Die Ressourcenabhängigkeit sank um 20%.",
            "E5-5: Lebenszyklusanalyse zentraler Produkte und Dienstleistungen": "Lebenszyklusanalysen wurden für 5 Hauptprodukte durchgeführt.",
            "S1-1: Demografie der Belegschaft (Alter, Geschlecht, Diversitätskennzahlen)": "Die Belegschaft umfasst 60% Männer und 40% Frauen, Durchschnittsalter 35 Jahre.",
            "S1-2: Arbeitsplatzsicherheit und Beschäftigungsverträge (befristet vs. unbefristet)": "95% der Beschäftigten haben unbefristete Verträge.",
            "S1-3: Unfallrate und Arbeitsschutzmaßnahmen": "Unfallrate sank auf 0,5 Unfälle pro 100 Mitarbeiter.",
            "S1-4: Schulung, Weiterqualifizierung und berufliche Entwicklung": "Durchschnittlich 40 Stunden Weiterbildung pro Mitarbeiter pro Jahr.",
            "S1-5: Vergütung, Zusatzleistungen und Lohngleichheit": "Lohngleichheit von 100% zwischen männlichen und weiblichen Mitarbeitern.",
            "S2-1: Sorgfaltspflicht in Bezug auf Menschenrechte und Risikominderung": "Wir haben Menschenrechtsprüfungen in 100% unserer Lieferkette durchgeführt.",
            "S2-2: Faire Löhne und Arbeitsbedingungen in der Lieferkette": "Faire Arbeitsbedingungen sind bei 90% unserer Lieferanten gewährleistet.",
            "S2-3: Gesundheits- und Sicherheitsvorfälle in Zulieferbetrieben": "Es gab 3 Gesundheits- und Sicherheitsvorfälle bei Zulieferern.",
            "S2-4: Risikoabschätzungen für Kinder- und Zwangsarbeit": "Keine Fälle von Kinder- oder Zwangsarbeit gefunden.",
            "S2-5: ESG-Compliance-Überwachung von Lieferanten": "ESG-Compliance-Rate unserer Lieferanten liegt bei 95%.",
            "S3-1: Gemeinschaftsbeteiligung und Wirkungsmessung": "Wir unterstützten lokale Gemeinschaften mit Investitionen von 1 Mio. Euro.",
            "S3-2: Soziale Investitionen und Beiträge zur lokalen Entwicklung": "Wir investierten 500.000 Euro in soziale Projekte.",
            "S3-3: Menschenrechtsauswirkungen auf betroffene Gemeinschaften": "Keine negativen Menschenrechtsauswirkungen in Gemeinschaften dokumentiert.",
            "S3-4: Zugang zu grundlegenden Dienstleistungen und Infrastruktur": "100% Zugang zu grundlegenden Dienstleistungen gewährleistet.",
            "S3-5: Konsultationsprozesse mit Stakeholdern": "Regelmäßige Konsultationen mit Stakeholdern, 4 Treffen pro Jahr.",
            "S4-1: Produktsicherheitsvorfälle und Einhaltung gesetzlicher Vorschriften": "Produktsicherheitsvorfälle reduzierten sich auf 2 Vorfälle pro Jahr.",
            "S4-2: Schutz der Verbraucherdaten und Datenschutzmaßnahmen": "Datenschutzverletzungen wurden um 90% reduziert.",
            "S4-3: Ethik im Marketing und faire Werbung": "100% Einhaltung ethischer Marketingstandards.",
            "S4-4: Mechanismen zur Bearbeitung von Kundenbeschwerden": "Durchschnittliche Bearbeitungszeit für Beschwerden sank auf 24 Stunden.",
            "S4-5: Barrierefreiheit und Inklusion von Produkten und Dienstleistungen": "Barrierefreiheit wurde für 90% unserer Produkte erreicht.",
            "G1-1: Struktur, Zusammensetzung und Diversität des Vorstands": "Vorstandsdiversität erhöhte sich auf 40% weibliche Mitglieder.",
            "G1-2: Vergütung von Führungskräften, Anreize und Transparenz": "Transparente Offenlegung der Vergütung für 100% der Führungskräfte.",
            "G1-3: Ethikrichtlinien und Maßnahmen zur Korruptionsbekämpfung": "Korruptionsfälle sanken auf null durch Ethikrichtlinien.",
            "G1-4: Schutz von Hinweisgebern, Meldekanäle und Compliance": "Meldekanäle für Hinweisgeber verzeichneten 50 Meldungen.",
            "G1-5: Risikomanagementrahmen, ESG-Aufsicht und interne Kontrollen": "Der ESG-Risikomanagementrahmen deckt alle Unternehmensbereiche ab.",
            "G1-6: ESG-Governance-Verantwortlichkeiten auf Vorstands- und Führungsebene": "ESG-Verantwortlichkeiten wurden in 100% der Führungsebene definiert.",
        }

        results = classify_text(list(texts_examples.values()))
        self.assertEqual(len(results), len(texts_examples))

        for i, (category, text) in enumerate(texts_examples.items()):
            self.assertEqual(results[i]["best_category"], category)
            self.assertTrue(0 <= results[i]["best_similarity"] <= 1)


if __name__ == "__main__":
    unittest.main()
