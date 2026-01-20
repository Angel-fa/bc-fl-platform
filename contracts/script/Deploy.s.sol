// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import "../src/ReceiptRegistry.sol";


contract DeployReceiptRegistry is Script {
    function run() external {
        vm.startBroadcast();
        new ReceiptRegistry();
        vm.stopBroadcast();
    }
}

"""
Αυτό το αρχείο το χρησιμοποιούμε όταν θέλουμε να «ανεβάσουμε» στο blockchain μόνο το συμβόλαιο ReceiptRegistry.
Δηλαδή, το κομμάτι της πλατφόρμας που κρατάει αποδείξεις ενεργειών (audit trail): ποιος έκανε τι και πότε,
χωρίς να αποθηκεύονται ευαίσθητα δεδομένα.
Στο σημείο new ReceiptRegistry(); ουσιαστικά λέμε:
«δημιούργησε ένα καινούργιο smart contract στο blockchain». Το vm.startBroadcast() και
vm.stopBroadcast() απλώς ανοίγουν και κλείνουν τη διαδικασία αποστολής της συναλλαγής.
Αυτό το script είναι χρήσιμο όταν θέλουμε απλή δοκιμή ή απομονωμένο deploy,
π.χ. μόνο για έλεγχο audit μηχανισμών.

"""