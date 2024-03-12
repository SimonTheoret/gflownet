# Notes
- Sans implémenter get_parents:
    - il faut utiliser trajectory balance

- intermediary reward:
    - il faut utiliser forward looking loss
    - cette loss utilises get_parents

- state2policy:
    - Fonction qui prend l'état et le donne au NN

- Choisir un nombre fini de mouvements
- Reward:
    - Commencer avec uniform reward
    - garder le score proche de 0

