<template>
        <div class=""> 
         <router-link to="/">Home</router-link> <a> </a>
        <router-link to="/about">About</router-link>
    </div>
    <router-view></router-view>

  </template>

<script>
// @ is an alias to /src
// import Header from '@/components/Header.vue'

export default {
  name: 'Nav',
  components: {
    // Header
  }
}
</script>