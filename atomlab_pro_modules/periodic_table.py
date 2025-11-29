
# periodic_table.py
# Contains full periodic table (Z, symbol, name, period, group, IE, EN, covalent radius)
PERIODIC = {
1: {'symbol':'H','name':'Hydrogen','period':1,'group':1,'IE':13.5984,'EN':2.20,'rcov':31},
2: {'symbol':'He','name':'Helium','period':1,'group':18,'IE':24.5874,'EN':None,'rcov':28},
3: {'symbol':'Li','name':'Lithium','period':2,'group':1,'IE':5.3917,'EN':0.98,'rcov':128},
4: {'symbol':'Be','name':'Beryllium','period':2,'group':2,'IE':9.3227,'EN':1.57,'rcov':96},
5: {'symbol':'B','name':'Boron','period':2,'group':13,'IE':8.298,'EN':2.04,'rcov':84},
6: {'symbol':'C','name':'Carbon','period':2,'group':14,'IE':11.2603,'EN':2.55,'rcov':76},
7: {'symbol':'N','name':'Nitrogen','period':2,'group':15,'IE':14.5341,'EN':3.04,'rcov':71},
8: {'symbol':'O','name':'Oxygen','period':2,'group':16,'IE':13.6181,'EN':3.44,'rcov':66},
9: {'symbol':'F','name':'Fluorine','period':2,'group':17,'IE':17.4228,'EN':3.98,'rcov':57},
10:{'symbol':'Ne','name':'Neon','period':2,'group':18,'IE':21.5645,'EN':None,'rcov':58},
11:{'symbol':'Na','name':'Sodium','period':3,'group':1,'IE':5.1391,'EN':0.93,'rcov':166},
12:{'symbol':'Mg','name':'Magnesium','period':3,'group':2,'IE':7.6462,'EN':1.31,'rcov':141},
13:{'symbol':'Al','name':'Aluminium','period':3,'group':13,'IE':5.9858,'EN':1.61,'rcov':121},
14:{'symbol':'Si','name':'Silicon','period':3,'group':14,'IE':8.1517,'EN':1.90,'rcov':111},
15:{'symbol':'P','name':'Phosphorus','period':3,'group':15,'IE':10.4867,'EN':2.19,'rcov':107},
16:{'symbol':'S','name':'Sulfur','period':3,'group':16,'IE':10.3600,'EN':2.58,'rcov':105},
17:{'symbol':'Cl','name':'Chlorine','period':3,'group':17,'IE':12.9676,'EN':3.16,'rcov':102},
18:{'symbol':'Ar','name':'Argon','period':3,'group':18,'IE':15.7596,'EN':None,'rcov':106},
# ... fill up to 118 with commonly available values (truncated for brevity in this template)
}

def get_element(Z):
    return PERIODIC.get(Z, None)
