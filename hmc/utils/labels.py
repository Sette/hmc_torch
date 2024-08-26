

def get_structure(genres_id, df_genres):
    def get_from_df(genre_id, df_genres, output=[]):
        if genre_id != 0:
            parent_genre = df_genres[df_genres["genre_id"]
                                     == genre_id].parent.values[0]
            output.append(genre_id)
            get_from_df(parent_genre, df_genres, output=output)
            return output
    output_list = []
    for genre_id in genres_id:
        output_list.append(get_from_df(genre_id, df_genres, output=[]))
    return output_list