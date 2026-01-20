// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import "../src/ReceiptRegistry.sol";
import "../src/ConsentRegistry.sol";

contract DeployAll is Script {
    function run() external {
        vm.startBroadcast();

        ReceiptRegistry rr = new ReceiptRegistry();
        ConsentRegistry cr = new ConsentRegistry();

        vm.stopBroadcast();

        console2.log("ReceiptRegistry:", address(rr));
        console2.log("ConsentRegistry:", address(cr));
    }
}

"""
Αυτό το αρχείο το χρησιμοποιούμε όταν θέλουμε να «ανεβάσουμε» στο blockchain μόνο το συμβόλαιο ReceiptRegistry.
Δηλαδή, το κομμάτι της πλατφόρμας που κρατάει αποδείξεις ενεργειών (audit trail): ποιος έκανε τι και πότε,
χωρίς να αποθηκεύονται ευαίσθητα δεδομένα.
Στο σημείο new ReceiptRegistry(); ουσιαστικά λέμε: «δημιούργησε ένα καινούργιο smart contract στο blockchain».
Το vm.startBroadcast() και vm.stopBroadcast() απλώς ανοίγουν και κλείνουν τη διαδικασία αποστολής της συναλλαγής.
 π.χ. για έλεγχο audit μηχανισμών.


--
"""
Το Deploy.s.sol χρησιμοποιείται για μεμονωμένο deploy του audit registry,
ενώ το DeployAll.s.sol για πλήρη αρχικοποίηση των on-chain μηχανισμών audit και consent.
Τα χρώματα στον editor σχετίζονται μόνο με το Git και όχι με τη λογική του συστήματος
"""