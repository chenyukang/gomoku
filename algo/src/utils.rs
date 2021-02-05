pub fn opponent(player: u8) -> u8 {
    match player {
        1 => 2,
        2 => 1,
        _ => panic!("error: {}", player),
    }
}
