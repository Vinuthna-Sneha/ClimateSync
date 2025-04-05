// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract GreenScore {
    mapping(address => uint256) public scores;

    event ScoreUpdated(address indexed user, uint256 newScore);

    function updateScore(uint256 points) public {
        scores[msg.sender] += points;
        emit ScoreUpdated(msg.sender, scores[msg.sender]);
    }

    function getScore(address user) public view returns (uint256) {
        return scores[user];
    }
}
